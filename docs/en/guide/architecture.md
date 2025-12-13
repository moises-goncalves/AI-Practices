# Architecture

System architecture design of AI-Practices.

## Progressive Learning Framework

```
┌─────────────────────────────────────────────────────────────┐
│                Progressive Learning Framework                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐        │
│   │ Theory │──▶│  Impl  │──▶│Framework│──▶│Practice│        │
│   │ First  │   │ Scratch│   │ Master  │   │Project │        │
│   └────────┘   └────────┘   └────────┘   └────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
Phase 1: Foundation
└── 01 Foundations

Phase 2: Core
├── 02 Neural Networks
├── 03 Computer Vision
└── 04 Sequence Models

Phase 3: Advanced
├── 05 Advanced Topics
├── 06 Generative Models
└── 07 Reinforcement Learning

Phase 4: Practice
└── 09 Practical Projects
```

## Directory Structure

```
AI-Practices/
├── 01-foundations/           # ML Foundations
├── 02-neural-networks/       # Neural Networks
├── 03-computer-vision/       # Computer Vision
├── 04-sequence-models/       # Sequence Models
├── 05-advanced-topics/       # Advanced Topics
├── 06-generative-models/     # Generative Models
├── 07-reinforcement-learning/# Reinforcement Learning
├── 08-theory-notes/          # Theory Notes
├── 09-practical-projects/    # Projects
└── utils/                    # Utilities
```

## Technology Choices

| Use Case | Primary | Alternative |
|:---------|:--------|:------------|
| Prototyping | TensorFlow/Keras | PyTorch |
| Research | PyTorch | JAX |
| Production | TensorFlow | ONNX |
| NLP | Transformers | spaCy |
| Tabular | XGBoost/LightGBM | CatBoost |
