# VoiceSteer: Emotional Control Infrastructure for Voice AI

**Never let your autonomous sales agent lose its temper.**

VoiceSteer is a B2B SaaS template and robust deployment framework for applying Dynamic Concept Steering to real-time Voice AI applications. Designed specifically for Call Centers, B2B Sales, and high-stakes customer support, VoiceSteer integrates cleanly with modern voice platforms (like Vapi.ai, Bland AI, Retell) via a low-latency websocket proxy.

Using the ERIS architecture underneath, VoiceSteer detects unwanted emotional states deep within the latent activations of your LLM and dynamically injects steering vectors to correct course *before* the bot speaks its next word.

## Features

- **Real-Time Latent Probing:** Monitor internal states for frustration, confusion, or aggressive sales tactics.
- **Dynamic Course Correction:** Inject dampening vectors (-Frustration) or enhancing vectors (+Empathy) instantly into the inference stream.
- **Microsecond Latency:** Built for streaming voice audio; our proxy adds negligible overhead.
- **Pre-Trained Emotion Vectors:** Works out-of-the-box with standard open-weight Llama and Mistral models.

## Usage

Check out the `examples/` directory for simulations of real-world interventions, such as de-escalating an angry customer in `vapi_intervention.py`.
