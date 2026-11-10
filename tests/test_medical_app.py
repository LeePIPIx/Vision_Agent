import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from medical_app.api import create_app
from medical_app.imaging import label_statistics, slice_png
from medical_app.storage import Store
from medical_app.tasks import TaskRunner


def volumes(tmp_path, units="mm"):
    affine = np.diag([2., 3., 4., 1.])
    image = nib.Nifti1Image(np.zeros((6, 7, 8), np.int16), affine)
    image.header.set_xyzt_units(units)
    mask = nib.Nifti1Image(np.zeros(image.shape, np.uint8), affine)
    mask.header.set_xyzt_units(units)
    mask_data = mask.get_fdata().astype(np.uint8)
    mask_data[1:3, 2:4, 3:5] = 1  # 8 voxels × 24 mm³ = 0.192 ml.
    mask = nib.Nifti1Image(mask_data, affine, mask.header)
    a,b = tmp_path / "ct.nii.gz", tmp_path / "mask.nii.gz"
    nib.save(image, a); nib.save(mask, b)
    return a,b


def test_measurements_and_overlay(tmp_path):
    image,mask = volumes(tmp_path)
    stats = label_statistics(image,mask)
    row = stats["labels"][0]
    assert row["voxel_count"] == 8
    assert row["volume_ml"] == pytest.approx(.192)
    assert row["connected_regions_6"] == 1
    assert row["centroid_ras_mm"] == [3.,7.5,14.]
    png,count = slice_png(image,mask,4)
    assert png.startswith(b"\x89PNG") and count == 8


def test_space_and_units_cannot_be_guessed(tmp_path):
    image,mask = volumes(tmp_path, units="unknown")
    with pytest.raises(ValueError, match="空间单位"):
        label_statistics(image,mask)
    image,mask = volumes(tmp_path)
    bad = nib.load(mask)
    affine = bad.affine.copy(); affine[0,3] += 10
    nib.save(nib.Nifti1Image(bad.get_fdata(), affine, bad.header),mask)
    with pytest.raises(ValueError, match="空间坐标"):
        label_statistics(image,mask)


@pytest.fixture
def app(tmp_path):
    return create_app(Store(tmp_path / "runtime"), start_worker=False)


def register(client, name="owner"):
    response = client.post("/api/auth/register", json={"username":name, "password":"long-test-password"})
    assert response.status_code == 200
    return response.json()


def submit(client, session="session-one", **kw):
    response = client.post("/api/tasks", data={"payload":json.dumps({"session_id":session,"query":"你好",**kw})})
    assert response.status_code == 202, response.text
    return response.json()["task_id"]


def test_auth_csrf_and_task_isolation(app):
    with TestClient(app) as owner, TestClient(app) as other:
        assert owner.get("/api/tasks").status_code == 401
        register(owner); register(other, "other")
        task = submit(owner)
        assert other.get(f"/api/tasks/{task}").status_code == 404
        assert other.post(f"/api/tasks/{task}/cancel").status_code == 404
        assert owner.post(f"/api/tasks/{task}/cancel", headers={"Origin":"https://attacker.example"}).status_code == 403
        duplicate = owner.post("/api/tasks", data={"payload":json.dumps({"session_id":"session-one","query":"again"})})
        assert duplicate.status_code == 409
        assert owner.post(f"/api/tasks/{task}/cancel").status_code == 200
        assert owner.get(f"/api/tasks/{task}").json()["status"] == "cancelled"
        assert owner.post(f"/api/tasks/{task}/retry").status_code == 202


def test_bad_upload_and_large_request(app):
    with TestClient(app) as c:
        register(c)
        r=c.post("/api/tasks",data={"payload":json.dumps({"session_id":"s","query":"CT"})},files={"file":("bad.nii.gz",b"not nifti")})
        assert r.status_code == 400
        r=c.post("/api/tasks",content=b"x",headers={"content-length":str(300*1024**2)})
        assert r.status_code == 413
        assert not app.state.store.query("SELECT * FROM files")


def test_file_access_and_reproducible_report(app,tmp_path):
    image,mask = volumes(tmp_path)
    store=app.state.store
    with TestClient(app) as c, TestClient(app) as other:
        owner=register(c); register(other,"other")
        r=c.post("/api/tasks",data={"payload":json.dumps({"session_id":"ct","query":"分割","workflow":"ct_spleen"})},files={"file":("ct.nii.gz",image.read_bytes())})
        assert r.status_code == 202
        task_id=r.json()["task_id"]
        task=store.task(task_id)
        owned_mask=store.root / "test_mask.nii.gz"; owned_mask.write_bytes(mask.read_bytes())
        mask_file=store.register_file(owner["id"],task["case_id"],owned_mask,"mask")
        req=task["request"];req["mask_id"]=mask_file["id"]
        store.execute("UPDATE tasks SET request=?,status='running' WHERE id=?",(json.dumps(req),task_id))
        TaskRunner(store).run(task_id)
        result=c.get(f"/api/tasks/{task_id}").json()
        assert result["status"] == "completed",result
        result=result["result"]
        assert result["report_data"]["statistics"]["labels"][0]["volume_ml"] == .192
        fid=result["viewer"]["image_id"]
        assert other.get(f"/api/files/{fid}").status_code == 404
        assert c.get(f"/api/files/{fid}/slice").status_code == 200
        assert c.post(f"/api/tasks/{task_id}/review",json={"status":"reviewed","note":"已核对边界"}).status_code == 200
        assert len(c.get(f"/api/tasks/{task_id}/reviews").json()) == 1
        assert other.get(f"/api/tasks/{task_id}/reviews").status_code == 404
        events=c.get(f"/api/tasks/{task_id}/events").text
        assert "measuring" in events and "event: done" in events
        persisted=Store(store.root)
        assert persisted.task(task_id)["result"] == result
        assert len(json.loads(persisted.query("SELECT messages FROM sessions",one=True)["messages"])) == 2


def test_restart_marks_running_interrupted_and_preserves_events(app):
    with TestClient(app) as c:
        register(c);tid=submit(c)
        app.state.store.execute("UPDATE tasks SET status='running' WHERE id=?",(tid,))
    runner=TaskRunner(app.state.store)
    runner.start();runner.close()
    assert app.state.store.task(tid)["status"] == "interrupted"
    assert app.state.store.query("SELECT * FROM events WHERE task_id=?",(tid,))


def test_cancel_prevents_task_execution(app):
    with TestClient(app) as c:
        register(c);tid=submit(c)
        c.post(f"/api/tasks/{tid}/cancel")
        TaskRunner(app.state.store).run(tid)
        assert app.state.store.task(tid)["status"] == "cancelled"


def test_ct_upload_auto_routes_and_followup_uses_chat(app, tmp_path):
    image, _ = volumes(tmp_path)
    with TestClient(app) as client:
        register(client)
        response = client.post("/api/tasks", data={"payload": json.dumps({"session_id": "auto-ct", "query": "请分析"})}, files={"file": ("ct.nii.gz", image.read_bytes())})
        assert response.status_code == 202, response.text
        first = response.json()["task_id"]
        assert app.state.store.task(first)["request"]["workflow"] == "ct_spleen"
        client.post(f"/api/tasks/{first}/cancel")
        response = client.post("/api/tasks", data={"payload": json.dumps({"session_id": "auto-ct", "query": "解释之前的结果"})})
        assert response.status_code == 202, response.text
        assert app.state.store.task(response.json()["task_id"])["request"]["workflow"] == "chat"
