# LatentSec-Fin: Zero-Hallucination Guardrails for Financial AI Agents via Mechanistic Interpretability

LatentSec-Fin is a robust B2B SaaS template designed to provide ironclad compliance and security for Financial AI Agents. By leveraging the **ERIS** (Latent Space Drift Detection) architecture and Mechanistic Interpretability, LatentSec-Fin moves beyond brittle keyword-based filters and prompt engineering to actively monitor the internal "thoughts" (latent space representations) of Large Language Models.

## The Problem

Financial AI agents handle highly sensitive operations. Traditional security measures rely on prompt wrappers or external LLM evaluator agents, which are:
- Vulnerable to sophisticated prompt injection attacks.
- Slow, introducing significant latency.
- Focused on the *output* text rather than the *intent* of the model.

## The LatentSec-Fin Solution

LatentSec-Fin sits as a proxy between your application and your model. By attaching Sparse Autoencoders (SAEs) to the model (like Gemma Scope), we map complex financial concepts to specific, interpretable features in the model's latent representation. 

We can directly observe if the model is currently "thinking" about making an unauthorized trade, issuing a refund, or asking for a password—*before* it even begins to generate the response text.

### Key Capabilities

*   **Zero-Hallucination Intent Blocking**: Block actions based on the internal state, stopping malicious prompt injections that bypass traditional text filters.
*   **Real-time Semantic Auditing**: Monitor exactly which financial concepts (e.g., 'Financial Commitment', 'High Risk Trading') are active during generation.
*   **OpenAI-Compatible Proxy**: Drop-in replacement for OpenAI endpoints. Your agents don't need to change their code—just point the API base to LatentSec-Fin.

## Structure

*   **/api_gateway**: Contains `compliance_proxy.py`, the core proxy that intercepts requests and leverages ERIS to analyze latent drift against defined rules.
*   **/rules_engine**: Contains the YAML definitions linking abstract financial risks (like "Unauthorized Refund") to concrete SAE feature IDs discovered via Mechanistic Interpretability.
*   **/examples**: Demonstrations of LatentSec-Fin defending against real-world prompt injection attacks on banking agents.
