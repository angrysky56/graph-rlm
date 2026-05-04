
import asyncio
import unittest.mock as mock
import graph_rlm.backend.src.core.agent as agent_mod

async def main():
    print(f"Original: {agent_mod.protected_llm_generate}")
    
    with mock.patch("graph_rlm.backend.src.core.agent.protected_llm_generate", new_callable=mock.AsyncMock) as m:
        print(f"After patch: {agent_mod.protected_llm_generate}")
        agent = agent_mod.Agent()
        agent.db = mock.Mock()
        agent.db.query.return_value = []
        agent.final_result = "Final Result"
        
        m.return_value = "Validated"
        res = await agent._generate_validated_response("r1", "task")
        print(f"Called: {m.called}")
        print(f"Result: {res}")

if __name__ == "__main__":
    asyncio.run(main())
