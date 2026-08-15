"""
Gerenciador de Checkpoints Persistentes e Retomada de Operações Interrompidas no SLD.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from src.sld.utils.atomic import atomic_write_json
from src.sld.utils.files import ensure_directory


class OperationCheckpoint(BaseModel):
    """Representa o snapshot persistente de uma operação em lote."""
    operation_id: str
    operation_type: str  # "ingestion", "segmentation", "embeddings", "index_update"
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"  # "active", "interrupted", "completed", "failed", "cancelled"
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    current_item: str = ""
    last_completed_item: str = ""
    pending_items: List[str] = Field(default_factory=list)
    completed_item_keys: List[str] = Field(default_factory=list)
    active_configuration: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, str]] = Field(default_factory=list)


class CheckpointManager:
    """Gerencia a persistência e recuperação de checkpoints em manifests/checkpoints.json."""

    def __init__(self, manifests_dir: Path):
        self.manifests_dir = Path(manifests_dir).expanduser().resolve()
        ensure_directory(self.manifests_dir)
        self.checkpoints_path = self.manifests_dir / "checkpoints.json"

    def load_checkpoints_data(self) -> Dict[str, Any]:
        """Carregamento seguro dos dados de checkpoints do disco."""
        if not self.checkpoints_path.exists():
            return {"active_operation_id": None, "operations": {}}
        try:
            with open(self.checkpoints_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"active_operation_id": None, "operations": {}}

    def save_checkpoint(self, checkpoint: OperationCheckpoint, set_as_active: bool = True) -> None:
        """Salva atomicamente o estado de um checkpoint."""
        checkpoint.last_updated_at = datetime.now().isoformat()
        data = self.load_checkpoints_data()

        if "operations" not in data:
            data["operations"] = {}

        data["operations"][checkpoint.operation_id] = checkpoint.model_dump()
        if set_as_active and checkpoint.status in ["active", "interrupted"]:
            data["active_operation_id"] = checkpoint.operation_id
        elif data.get("active_operation_id") == checkpoint.operation_id and checkpoint.status in ["completed", "cancelled"]:
            data["active_operation_id"] = None

        atomic_write_json(self.checkpoints_path, data)

    def get_active_checkpoint(self) -> Optional[OperationCheckpoint]:
        """Retorna a operação atualmente ativa ou interrompida, se houver."""
        data = self.load_checkpoints_data()
        active_id = data.get("active_operation_id")

        if active_id and active_id in data.get("operations", {}):
            chk_dict = data["operations"][active_id]
            return OperationCheckpoint(**chk_dict)

        # Procura qualquer operação pendente com status 'active' ou 'interrupted'
        for chk_dict in data.get("operations", {}).values():
            if chk_dict.get("status") in ["active", "interrupted"]:
                return OperationCheckpoint(**chk_dict)

        return None

    def create_checkpoint(
        self,
        operation_type: str,
        total_items: int,
        all_item_keys: List[str],
        config: Dict[str, Any]
    ) -> OperationCheckpoint:
        """Cria e salva um novo checkpoint inicial para uma operação."""
        op_id = f"op_{operation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        chk = OperationCheckpoint(
            operation_id=op_id,
            operation_type=operation_type,
            status="active",
            total_items=total_items,
            completed_items=0,
            failed_items=0,
            current_item="",
            last_completed_item="",
            pending_items=list(all_item_keys),
            completed_item_keys=[],
            active_configuration=config
        )
        self.save_checkpoint(chk, set_as_active=True)
        return chk

    def update_item_success(
        self,
        chk: OperationCheckpoint,
        item_key: str,
        current_item_name: str = ""
    ) -> OperationCheckpoint:
        """Registra a conclusão bem-sucedida de um item no checkpoint."""
        chk.completed_items += 1
        chk.last_completed_item = item_key
        chk.current_item = current_item_name or item_key
        if item_key in chk.pending_items:
            chk.pending_items.remove(item_key)
        if item_key not in chk.completed_item_keys:
            chk.completed_item_keys.append(item_key)
        self.save_checkpoint(chk)
        return chk

    def update_item_error(
        self,
        chk: OperationCheckpoint,
        item_key: str,
        error_msg: str,
        current_item_name: str = ""
    ) -> OperationCheckpoint:
        """Registra uma falha em um item no checkpoint sem interromper o lote."""
        chk.failed_items += 1
        chk.current_item = current_item_name or item_key
        chk.errors.append({"item_key": item_key, "error": error_msg, "timestamp": datetime.now().isoformat()})
        if item_key in chk.pending_items:
            chk.pending_items.remove(item_key)
        self.save_checkpoint(chk)
        return chk

    def mark_completed(self, chk: OperationCheckpoint) -> None:
        """Marca a operação do checkpoint como concluída com sucesso."""
        chk.status = "completed"
        chk.pending_items = []
        self.save_checkpoint(chk, set_as_active=False)

    def mark_interrupted(self, chk: OperationCheckpoint) -> None:
        """Marca a operação do checkpoint como interrompida pelo usuário ou erro fatal."""
        chk.status = "interrupted"
        self.save_checkpoint(chk, set_as_active=True)

    def mark_cancelled(self, chk: OperationCheckpoint) -> None:
        """Marca a operação do checkpoint como cancelada explicitamente."""
        chk.status = "cancelled"
        self.save_checkpoint(chk, set_as_active=False)
