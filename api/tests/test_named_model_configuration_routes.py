"""Named model configurations never store or return a null ``tts_fallback``.

Dograh 1.51 rejects that key on a named configuration (``extra="forbid"``), so
a configuration without a voice fallback must stay readable by it after a
rollback.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.model_connections import router
from api.services.auth.depends import get_user_with_selected_organization

CONNECTION = "11111111-1111-4111-8111-111111111111"
BACKUP = "22222222-2222-4222-8222-222222222222"
SELECTION = {"provider_connection_uuid": CONNECTION, "settings": {}}
POLICY = {
    "version": 1,
    "rules": [
        {
            "condition": {"type": "error"},
            "target": {"provider_connection_uuid": BACKUP, "settings": {}},
        }
    ],
}


def specification(**extra):
    return {
        "version": 3,
        "mode": "pipeline",
        "llm": SELECTION,
        "stt": SELECTION,
        "tts": SELECTION,
        **extra,
    }


class InMemoryCatalog:
    """The catalog calls the routes make, keeping the rows exactly as written."""

    def __init__(self):
        self.rows = {}

    def add(self, configuration):
        row = SimpleNamespace(
            uuid=str(uuid4()),
            name="Voice",
            configuration=deepcopy(configuration),
            revision=1,
            is_active=True,
        )
        self.rows[row.uuid] = row
        return row

    async def create_named_model_configuration(
        self, organization_id, *, name, configuration
    ):
        row = self.add(configuration)
        row.name = name
        return row

    async def get_named_model_configuration(
        self, organization_id, configuration_uuid, active_only=True
    ):
        return self.rows.get(configuration_uuid)

    async def list_named_model_configurations(self, organization_id, active_only=True):
        return list(self.rows.values())

    async def update_named_model_configuration(
        self, organization_id, configuration_uuid, *, changes, expected_revision=None
    ):
        row = self.rows[configuration_uuid]
        for key, value in changes.items():
            setattr(row, key, deepcopy(value))
        row.revision += 1
        return row


@pytest.fixture
def catalog():
    return InMemoryCatalog()


@pytest.fixture
def client(catalog):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_user_with_selected_organization] = lambda: (
        SimpleNamespace(id=1, selected_organization_id=11, provider_id="user")
    )
    with (
        patch("api.routes.model_connections.db_client", catalog),
        patch(
            "api.routes.model_connections.resolve_inline_model_configuration",
            AsyncMock(),
        ),
        patch("api.routes.model_connections.get_default", AsyncMock(return_value=None)),
    ):
        yield TestClient(app)


@pytest.mark.parametrize("extra", [{}, {"tts_fallback": None}])
def test_create_without_a_voice_fallback_stores_and_returns_no_key(
    client, catalog, extra
):
    response = client.post(
        "/model-connections/model-configurations",
        json={"name": "Voice", "configuration": specification(**extra)},
    )
    assert response.status_code == 201
    (row,) = catalog.rows.values()
    assert "tts_fallback" not in row.configuration
    configuration = response.json()["configuration"]
    assert "tts_fallback" not in configuration
    # Only the new key is omitted: the rest of the shape is what it was.
    assert row.configuration["llm_fallback"] is None
    assert configuration["llm_fallback"] is None


@pytest.mark.parametrize("extra", [{}, {"tts_fallback": None}])
def test_update_without_a_voice_fallback_stores_and_returns_no_key(
    client, catalog, extra
):
    row = catalog.add(specification(tts_fallback=POLICY))
    response = client.patch(
        f"/model-connections/model-configurations/{row.uuid}",
        json={"configuration": specification(**extra), "revision": 1},
    )
    assert response.status_code == 200
    assert "tts_fallback" not in catalog.rows[row.uuid].configuration
    assert "tts_fallback" not in response.json()["configuration"]


def test_rows_stored_with_a_null_key_are_returned_without_it(client, catalog):
    row = catalog.add(specification(llm_fallback=None, tts_fallback=None))
    assert row.configuration["tts_fallback"] is None
    listed = client.get("/model-connections/model-configurations").json()
    fetched = client.get(f"/model-connections/model-configurations/{row.uuid}").json()
    assert "tts_fallback" not in listed[0]["configuration"]
    assert "tts_fallback" not in fetched["configuration"]


def test_a_voice_fallback_policy_round_trips(client, catalog):
    created = client.post(
        "/model-connections/model-configurations",
        json={"name": "Voice", "configuration": specification(tts_fallback=POLICY)},
    )
    assert created.status_code == 201
    (row,) = catalog.rows.values()
    assert row.configuration["tts_fallback"] == POLICY
    assert created.json()["configuration"]["tts_fallback"] == POLICY

    other = {**POLICY, "rules": [{**POLICY["rules"][0], "target": SELECTION}]}
    updated = client.patch(
        f"/model-connections/model-configurations/{row.uuid}",
        json={"configuration": specification(tts_fallback=other), "revision": 1},
    )
    assert updated.status_code == 200
    assert catalog.rows[row.uuid].configuration["tts_fallback"] == other
    assert updated.json()["configuration"]["tts_fallback"] == other
    fetched = client.get(f"/model-connections/model-configurations/{row.uuid}")
    assert fetched.json()["configuration"]["tts_fallback"] == other
