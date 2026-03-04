from .chunk_list_adjustment import adjustment_chunks
from .chunks_list_T01_14_trauma import trauma_chunks
from .chunks_list_E01_environment import environment_chunks
from .chunks_list_P01_P13_pediatric import pediatric_chunks
from .chunks_list_A01_A13_non_trauma import non_trauma_chunks

chunks_list = (
    # adjustment_chunks +
    trauma_chunks + environment_chunks + pediatric_chunks + non_trauma_chunks
)
