"""
Utilitários de manipulação de arquivos, diretórios e sanitização de caminhos.
"""

import re
import sys
import subprocess
import unicodedata
from pathlib import Path
from typing import List, Optional


def sanitize_filename(name: str) -> str:
    """
    Converte uma string em um nome de arquivo seguro para o sistema de arquivos.
    Remove acentos, caracteres especiais e limita espaços.
    """
    normalized = unicodedata.normalize("NFKD", name)
    ascii_bytes = normalized.encode("ASCII", "ignore")
    cleaned = ascii_bytes.decode("ascii")

    cleaned = re.sub(r'[^\w\s-]', '', cleaned).strip()
    cleaned = re.sub(r'[-\s]+', '_', cleaned)

    return cleaned if cleaned else "documento"


def validate_directory(dir_path: str) -> Path:
    """
    Valida se o caminho especificado é um diretório existente e acessível.
    Lança ValueError se for inválido.
    """
    path = Path(dir_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"O diretório especificado não existe: {dir_path}")
    if not path.is_dir():
        raise ValueError(f"O caminho especificado não é um diretório: {dir_path}")
    return path


def find_pdf_files(directory: Path, recursive: bool = False) -> List[Path]:
    """
    Busca arquivos `.pdf` em um diretório, podendo ser busca direta ou recursiva em subdiretórios.
    Retorna a lista de caminhos absolutos ordenados pelo nome do arquivo.
    """
    if recursive:
        pdfs = list(directory.rglob("*.pdf")) + list(directory.rglob("*.PDF"))
    else:
        pdfs = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))

    unique_pdfs = sorted(list(set(p.resolve() for p in pdfs)))
    return unique_pdfs


def ensure_directory(path: Path) -> Path:
    """Garante que o diretório exista, criando-o recursivamente se necessário."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_file_hash(file_path: Path) -> str:
    """Calcula o hash SHA-256 do arquivo."""
    import hashlib
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def open_folder_dialog(initial_dir: str = "", title: str = "Selecionar Diretório") -> str:
    """
    Abre a caixa de diálogo nativa do sistema operacional (macOS Finder / Windows File Explorer)
    para seleção de diretórios sem permitir digitação manual.
    """
    init_p = str(Path(initial_dir).resolve()) if initial_dir and Path(initial_dir).exists() else str(Path.home())

    if sys.platform == "darwin":
        script = f'''
        tell application "System Events"
            activate
            try
                set chosenFolder to choose folder with prompt "{title}" default location POSIX file "{init_p}"
                return POSIX path of chosenFolder
            on error
                return ""
            end try
        end tell
        '''
        try:
            res = subprocess.check_output(["osascript", "-e", script], text=True)
            return res.strip()
        except Exception:
            return ""
    elif sys.platform == "win32":
        try:
            ps_cmd = f"(New-Object -ComObject Shell.Application).BrowseForFolder(0, '{title}', 0, '{init_p}').Self.Path"
            res = subprocess.check_output(["powershell", "-Command", ps_cmd], text=True)
            return res.strip()
        except Exception:
            return ""
    else:
        try:
            res = subprocess.check_output(["zenity", "--file-selection", "--directory", f"--filename={init_p}"], text=True)
            return res.strip()
        except Exception:
            return ""
