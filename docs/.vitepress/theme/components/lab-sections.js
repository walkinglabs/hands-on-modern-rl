export const labSections = [
  {
    id: 'foundations',
    name: { zh: '基础与经典 RL', en: 'Foundations & classical RL' },
    ids: ['cartpole', 'gymnasium', 'bandit', 'gridworld']
  },
  {
    id: 'deep',
    name: { zh: '深度强化学习', en: 'Deep reinforcement learning' },
    ids: [
      'dqn',
      'reinforce',
      'pendulum',
      'jax',
      'vizdoom',
      'atari',
      'minecraft',
      'unity'
    ]
  },
  {
    id: 'advanced',
    name: { zh: '高级强化学习', en: 'Advanced reinforcement learning' },
    ids: ['minigrid', 'multiagent', 'selfplay', 'offline']
  },
  {
    id: 'alignment',
    name: { zh: '大模型对齐', en: 'Language model alignment' },
    ids: ['dpo', 'grpo', 'financial']
  },
  {
    id: 'agentic',
    name: { zh: 'Agentic RL', en: 'Agentic RL' },
    ids: ['agentic-tools', 'agentic-code', 'agentic-finqa', 'agentic-research']
  },
  {
    id: 'embodied',
    name: { zh: '世界模型与具身', en: 'World models & embodiment' },
    ids: ['worldmodel', 'maniskill', 'ai2thor']
  }
]

const legacySections = {
  intro: 'foundations',
  mdp: 'foundations',
  dqn: 'deep',
  pg: 'deep',
  ac: 'deep',
  ppo: 'deep',
  explore: 'advanced',
  selfplay: 'advanced',
  dpo: 'alignment',
  grpo: 'alignment',
  research: 'agentic'
}

export function resolveLabSection(id) {
  if (labSections.some((section) => section.id === id)) return id
  return Object.hasOwn(legacySections, id) ? legacySections[id] : 'all'
}
