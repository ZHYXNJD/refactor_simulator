"""
兼容性模块

这个文件作为兼容性层，将对 value_estimatior.sarsa 的导入重定向到 src/agents/
"""
from src.agents.sarsa import SarsaAgent

__all__ = ['SarsaAgent']