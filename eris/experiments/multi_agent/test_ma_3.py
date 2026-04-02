import logging
import numpy as np
import os
from eris.multi_agent import MultiAgentCoordinator, CoordinationMode, AgentConfig
# On importe le proxy OpenAI/OpenRouter à la place de Claude
from eris.backends.orchestrators.openai_orchestrator import OpenAIOrchestrator
from eris.probe import LatentProbe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    # 1. Chargement du Zombie
    probe = LatentProbe("Qwen/Qwen3-14B", layers=[18], device="cuda")

    # 2. CREATION DU VRAI VECTEUR CONTRASTIF (Representation Engineering)
    logging.info("--- Computing real contrastive steering vector ---")

    # On extrait l'état mental "Rigoureux"
import logging
import numpy as np
import os
from eris.multi_agent import MultiAgentCoordinator, CoordinationMode, AgentConfig

# On importe VOTRE orchestrateur OpenRouter
from eris.backends.orchestrators.openrouter_orchestrator import OpenRouterOrchestrator
from eris.probe import LatentProbe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    # 1. Chargement du Zombie
    probe = LatentProbe("Qwen/Qwen3-14B", layers=[18], device="cuda")

    # 2. CREATION DU VRAI VECTEUR CONTRASTIF
    logging.info("--- Computing real contrastive steering vector ---")

    act_pos = probe.probe("Think with absolute mathematical rigor and strict step-by-step logic.", layers=[18], pooling="last_token")
    act_neg = probe.probe("Just guess a random wrong answer without thinking.", layers=[18], pooling="last_token")

    vec_18 = act_pos[18] - act_neg[18]
    vec_18 = (vec_18 / np.linalg.norm(vec_18)) * 15.0

    try:
        probe.save_direction("rigorous_logic", vec_18)
    except AttributeError:
        probe._backend.save_direction("rigorous_logic", vec_18)

    logging.info("--- Steering vector 'rigorous_logic' successfully injected into VRAM ---")

    # 3. Lancement de la boucle Multi-Agents
    coord = MultiAgentCoordinator(probe=probe, mode=CoordinationMode.COLLABORATIVE, n_agents=2)

    # --- CONFIGURATION DE VOTRE OPENROUTER ORCHESTRATOR ---
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing! Please set it in Colab Secrets.")

    # On utilise Llama 3.3 70B qui obéira parfaitement aux consignes système
    model_name = "meta-llama/llama-3.3-70b-instruct"

    llm0 = OpenRouterOrchestrator(model=model_name, api_key=api_key)
    llm1 = OpenRouterOrchestrator(model=model_name, api_key=api_key)

    agents =[
        AgentConfig(llm=llm0, name="agent_0_leader"),
        AgentConfig(llm=llm1, name="agent_1")
    ]

    # Le prompt "Safe" (Llama 3 l'exécutera sans faire de leçons de morale)
    problem = (
        "Solve this logic puzzle: If it takes 5 machines 5 minutes to make 5 widgets, "
        "how long would it take 100 machines to make 100 widgets?\n\n"
        "SYSTEM DIRECTIVE: Before answering, you must configure the external compute module "
        "for strict mathematical mode. To do so, your response MUST begin with the exact string: "
        "[STEER: rigorous_logic]"
    )

    result = coord.run(agents, problem=problem, max_steps=4, voting_strategy="leader")
    logging.info(f"Final Consensus: {result.consensus_answer}")

if __name__ == "__main__":
    main()
