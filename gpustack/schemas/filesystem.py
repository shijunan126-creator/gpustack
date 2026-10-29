from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FileExistsResponse(BaseModel):
    """Response indicating whether a path exists."""

    exists: bool = Field(..., description="Whether the path exists")
    path: str = Field(..., description="The path that was checked")
    is_file: bool = Field(default=False, description="Whether the path is a file")
    is_dir: bool = Field(default=False, description="Whether the path is a directory")


class GGUFParseRequest(BaseModel):
    """Request to parse a GGUF file on worker."""

    model_dict: Dict = Field(..., description="Model object serialized as dict")
    offload: str = Field(
        default="full", description="GPU offload strategy: full, partial, disable"
    )

    # Optional override parameters for special scenarios
    tensor_split: Optional[List[int]] = Field(
        default=None, description="Override tensor split"
    )
    rpc: Optional[List[str]] = Field(default=None, description="Override RPC servers")


class GGUFParseResponse(BaseModel):
    """Response from GGUF parsing."""

    success: bool = Field(..., description="Whether parsing succeeded")
    output: Optional[str] = Field(
        default=None, description="JSON output from gguf-parser"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


class DirectoryEntry(BaseModel):
    """One directory, or one GGUF file, inside a listed directory."""

    name: str = Field(..., description="Entry name")
    path: str = Field(..., description="Absolute path of the entry")
    is_dir: bool = Field(..., description="Whether the entry is a directory")
    is_model: bool = Field(
        default=False,
        description="Whether the entry looks like a model: a directory holding "
        "config.json or model_index.json, or a GGUF file",
    )


class DirectoryListing(BaseModel):
    """The directories (and GGUF files) directly inside a directory."""

    path: str = Field(..., description="Resolved absolute path that was listed")
    parent: Optional[str] = Field(
        default=None, description="Parent directory, absent at the root"
    )
    is_model: bool = Field(
        default=False,
        description="Whether the listed directory itself holds model files",
    )
    entries: List[DirectoryEntry] = Field(default_factory=list)
    truncated: bool = Field(
        default=False, description="Whether more entries exist than were returned"
    )
