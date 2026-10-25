from __future__ import annotations

import io
import json
from threading import Event
from time import monotonic, sleep

from spittrack.web.app import create_app


def test_demo_missing_model_and_invalid_uploads(tmp_path):
    client = create_app(project_root=tmp_path).test_client()
    assert client.get('/').status_code == 200
    assert b'Final model provenance is unavailable' in client.get('/').data
    assert client.post('/api/runs').status_code == 400
    assert client.post('/api/runs', data={'video': (io.BytesIO(b'x'), 'bad.txt')}).status_code == 400
    assert client.get('/api/runs/not-a-run').status_code == 404


def test_demo_background_failure_is_reported_and_artifacts_restricted(tmp_path):
    model_dir = tmp_path / 'outputs' / 'final_model'
    model_dir.mkdir(parents=True)
    (model_dir / 'final_model_provenance.json').write_text(json.dumps({'checkpoint_path': 'final.pt'}))
    started = Event()
    finish = Event()

    def failing_runner(*args):
        started.set()
        assert finish.wait(5)
        raise RuntimeError('Invalid video')

    client = create_app(project_root=tmp_path, runner=failing_runner).test_client()
    response = client.post('/api/runs', data={'video': (io.BytesIO(b'video'), 'real.mp4')})
    assert response.status_code == 202
    job_id = response.json['id']
    assert started.wait(5)
    assert client.post('/api/runs', data={'video': (io.BytesIO(b'video'), 'other.mp4')}).status_code == 409
    finish.set()
    deadline = monotonic() + 5
    while monotonic() < deadline:
        status = client.get(f'/api/runs/{job_id}').json
        if status['state'] == 'failed':
            break
        sleep(0.02)
    assert status['state'] == 'failed'
    assert client.get(f'/runs/{job_id}/files/final_model_provenance.json').status_code == 404
