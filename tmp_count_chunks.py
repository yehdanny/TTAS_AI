# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data", "chunks"))

from chunks_list_A01_A13_non_trauma import non_trauma_chunks
from chunks_list_P01_P13_pediatric import pediatric_chunks
from chunks_list_T01_14_trauma import trauma_chunks
from chunks_list_E01_environment import environment_chunks
from chunk_list_adjustment import adjustment_chunks

print(f"成人非外傷: {len(non_trauma_chunks)}")
print(f"兒童: {len(pediatric_chunks)}")
print(f"外傷: {len(trauma_chunks)}")
print(f"環境: {len(environment_chunks)}")
print(f"調節變數(總表): {len(adjustment_chunks)}")
print(f"合計: {len(non_trauma_chunks)+len(pediatric_chunks)+len(trauma_chunks)+len(environment_chunks)+len(adjustment_chunks)}")

# 顯示第一個 chunk 確認結構
print("\n[成人非外傷 chunk 0 keys]:", list(non_trauma_chunks[0].keys()))
print("[調節 chunk 0 keys]:", list(adjustment_chunks[0].keys()))
