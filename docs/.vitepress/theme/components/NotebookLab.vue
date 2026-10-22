<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { withBase } from 'vitepress'
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronDown,
  Cpu,
  FlaskConical,
  Grid2X2,
  List,
  NotebookPen,
  X
} from 'lucide-vue-next'
import catalog from './lab-catalog.json'
import englishLabs from './lab-english.json'
import { labSections, resolveLabSection } from './lab-sections.js'

const props = defineProps({ locale: { type: String, default: 'zh' } })
const english = props.locale === 'en'
const copy = english
  ? {
      heading: 'Lab',
      intro: 'Turn the ideas in the book into an experiment of your own.',
      subintro: 'Choose a Notebook and open it in ModelScope or Colab.',
      all: 'All experiments',
      chapterFilter: 'Filter by section',
      notebooks: 'Notebooks',
      browse: 'Browse experiments',
      cards: 'Cards',
      list: 'Contents',
      cardLabel: 'Card view',
      listLabel: 'Contents view',
      shelfLabel: 'Notebook experiments; swipe horizontally to browse',
      details: 'Experiment details',
      environment: 'Environment preview',
      schematic: 'Task illustration',
      previous: 'Previous experiment',
      next: 'Next experiment',
      swipe: 'Swipe to discover the next experiment',
      usage: 'Read, then run.',
      usageDesc:
        'Every experiment links to its companion chapter. Read the method, then inspect the training curves and policy replays.',
      start: 'Start with the first experiment',
      planned: 'More experiments to come',
      preparing: 'being prepared',
      footnote:
        'Open a Notebook to run it, or view saved curves and policy replays from its result page.',
      close: 'Close experiment details',
      bookPreview: 'Environment preview from the book',
      experimentPreview: 'Local experiment result',
      verification: 'Local execution record',
      method: 'Method',
      resource: 'Resources',
      kind: 'Experiment type',
      result: 'What you will get',
      openModelscope: 'Open in ModelScope',
      openColab: 'Open in Colab',
      chapter: 'Companion chapter',
      source: 'Notebook source',
      savedResult: 'View results',
      runtime: 'Running and reproducing the experiment',
      runtimeDesc:
        'Use the resource profile above and run the cells in order to generate training curves and policy replays.',
      runtimeExecutedDesc:
        'The Notebook includes outputs from its completed CPU run. Open the result page for plots, metrics and execution outputs.',
      review: 'View',
      cpu: 'CPU'
    }
  : {
      heading: '实验室',
      intro: '把书中的原理，变成一次亲手完成的实验。',
      subintro: '选择 Notebook，在 ModelScope 或 Colab 中开始。',
      all: '全部实验',
      chapterFilter: '按内容分类',
      notebooks: '个 Notebook',
      browse: '浏览方式',
      cards: '卡片',
      list: '目录',
      cardLabel: '卡片浏览',
      listLabel: '目录浏览',
      shelfLabel: 'Notebook 实验卡片，可左右滑动',
      details: '实验说明',
      environment: '环境预览',
      schematic: '任务示意',
      previous: '上一个实验',
      next: '下一个实验',
      swipe: '左右滑动，发现下一个实验',
      usage: '边读，边运行。',
      usageDesc:
        '每个实验都连接书中的配套章节。先读原理，再观察训练曲线与策略回放。',
      start: '从第一个实验开始',
      planned: '接下来，还会有这些实验',
      preparing: '项整理中',
      footnote:
        '打开 Notebook 运行实验，也可以直接查看已保存的曲线与策略回放。',
      close: '关闭实验说明',
      bookPreview: '书中环境预览',
      experimentPreview: '本地实验结果',
      verification: '本地运行记录',
      method: '方法',
      resource: '运行资源',
      kind: '实验类型',
      result: '你会得到什么',
      openModelscope: '在 ModelScope 打开',
      openColab: '在 Colab 打开',
      chapter: '配套章节',
      source: 'Notebook 源文件',
      savedResult: '查看运行结果',
      runtime: '运行与复现说明',
      runtimeDesc:
        '按上方资源需求准备环境，再依次运行单元格，生成训练曲线与策略回放。',
      runtimeExecutedDesc:
        'Notebook 已保存本次 CPU 运行的输出；结果页包含图表、指标与运行输出。',
      review: '查看',
      cpu: 'CPU'
    }

const chapters = labSections.map((section) => ({
  ...section,
  name: section.name[english ? 'en' : 'zh']
}))
const kindNames = {
  真实训练: 'Training',
  数值实验: 'Numerical experiment',
  机制演示: 'Mechanism demonstration'
}
const deviceNames = {
  'GPU · 原生模拟器': 'GPU · Native simulator',
  'GPU · Linux 场景': 'GPU · Linux scene',
  'GPU · 图形后端': 'GPU · Graphics backend',
  '单卡 GPU': 'Single GPU'
}
const methodNames = {
  经典控制: 'Classical control',
  '视觉 PPO': 'Visual PPO',
  策略梯度: 'Policy gradient',
  'PPO · 探索': 'PPO · Exploration',
  多智能体: 'Multi-agent',
  机器人控制: 'Robot control',
  视觉导航: 'Visual navigation',
  '自博弈 · CFR+': 'Self-play · CFR+'
}
const info = {
  'agentic-tools': [
    'Multi-Turn',
    '多轮工具与信用分配',
    'REINFORCE',
    'agentic-tools-result.png'
  ],
  'agentic-code': [
    'Code Agent',
    '让代码 Agent 修复并通过测试',
    'REINFORCE',
    'agentic-code-result.png'
  ],
  'agentic-finqa': [
    'FinQA Agent',
    '让 Agent 查表、计算并作答',
    'REINFORCE',
    'agentic-finqa-result.png'
  ],
  'agentic-research': [
    'Research',
    '让研究 Agent 检索并给出证据',
    'Feature policy',
    'agentic-research-result.png'
  ],
  bandit: [
    'Bandit',
    '在尝试中找到更好的动作',
    'UCB · ε-greedy',
    'bandit-result.png'
  ],
  gridworld: [
    'GridWorld',
    '从价值表走到终点',
    'Q-learning',
    'gridworld-result.png'
  ],
  dqn: [
    'DQN · Double DQN',
    '拆开动作选择和价值评价',
    'DQN · Double DQN',
    'dqn-result.gif'
  ],
  reinforce: [
    'REINFORCE',
    '用状态基线检查策略梯度',
    'REINFORCE',
    'reinforce-result.gif'
  ],
  pendulum: ['Pendulum', '学习连续力矩控制', 'A2C', 'pendulum-result.gif'],
  dpo: ['DPO', '训练一个微型语言模型的偏好', 'DPO · GRU', 'dpo-result.png'],
  grpo: ['GRPO · RLVR', '逐项检查奖励和优势', 'GRPO', 'grpo-result.png'],
  financial: [
    'Tool Calling',
    '用模拟执行结果更新策略',
    'GRPO · GRU',
    'financial-result.png'
  ],
  worldmodel: [
    'World Model',
    '预测动作后果，再规划',
    'Model-based RL',
    'worldmodel-result.png'
  ],
  offline: [
    'Offline RL',
    '检查数据之外的价值估计',
    'CQL-style · BC',
    'offline-result.png'
  ],
  cartpole: ['CartPole', '让小车保持平衡', 'PPO', 'cartpole.gif'],
  gymnasium: ['Gymnasium', '尝试不同的控制环境', '经典控制', 'gymnasium.gif'],
  vizdoom: ['ViZDoom', '从第一人称画面做决策', 'DQN', 'vizdoom.gif'],
  atari: ['Atari', '让像素变成动作', 'DQN · CNN', 'atari.gif'],
  minecraft: ['Minecraft', '训练第一人称视觉策略', '视觉 PPO', 'minecraft.jpg'],
  jax: ['JAX · MinAtar', '向量化训练像素小游戏', '策略梯度', null],
  unity: ['Unity ML-Agents', '在物理场景里学习控制', 'PPO', 'unity.webp'],
  minigrid: ['MiniGrid', '在迷宫里发现奖励', 'PPO · 探索', null],
  multiagent: ['PettingZoo', '让多个智能体学会合作', '多智能体', null],
  maniskill: ['ManiSkill', '在仿真中训练机器人', '机器人控制', 'maniskill.jpg'],
  ai2thor: ['AI2-THOR', '在房间里寻找目标', '视觉导航', 'ai2thor.jpg'],
  selfplay: ['OpenSpiel', '通过对手发现更好的策略', '自博弈 · CFR+', null]
}
const labs = chapters.flatMap((ch) =>
  ch.ids.map((id) => {
    const original = catalog.labs.find((l) => l.id === id)
    const lab = english
      ? {
          ...original,
          ...englishLabs[id],
          kind: kindNames[original.kind],
          device: deviceNames[original.device] || original.device
        }
      : original
    const presentation = info[id]
    return {
      ...lab,
      path: original.paths?.[english ? 'en' : 'zh'] || original.path,
      chapterId: ch.id,
      chapterName: ch.name,
      title: presentation?.[0] || lab.en.split(' · ')[0],
      subtitle: english ? lab.name : presentation?.[1] || lab.name,
      method: english
        ? methodNames[presentation?.[2]] || presentation?.[2] || lab.kind
        : presentation?.[2] || lab.kind,
      image: presentation?.[3]
        ? withBase('/lab-previews/' + presentation[3])
        : null,
      resultUrl: original.results?.[english ? 'en' : 'zh']
        ? withBase('/' + original.results[english ? 'en' : 'zh'])
        : null
    }
  })
)
const existing = labs.filter((l) => l.path)
const planned = labs.filter((l) => !l.path)
const availableChapters = chapters.filter((ch) =>
  existing.some((l) => l.chapterId === ch.id)
)
const selectedChapter = ref('all')
const view = ref('cards')
const chapterMenu = ref(null)
const menuAbove = ref(false)
const menuHeight = ref(420)
const viewport = ref(null)
const modal = ref(null)
const active = ref(null)
const start = ref(0)
const end = ref(0)
const canPrev = ref(false)
const canNext = ref(true)
const filtered = computed(() =>
  existing.filter(
    (l) =>
      selectedChapter.value === 'all' || l.chapterId === selectedChapter.value
  )
)
const chapterLabel = computed(
  () =>
    availableChapters.find((ch) => ch.id === selectedChapter.value)?.name ||
    copy.all
)
const visibleChapters = computed(() =>
  availableChapters.filter((ch) =>
    filtered.value.some((l) => l.chapterId === ch.id)
  )
)
const repository = 'https://github.com/walkinglabs/hands-on-modern-rl/'
const platformUrl = (lab, platform) =>
  platform === 'colab'
    ? 'https://colab.research.google.com/github/walkinglabs/hands-on-modern-rl/blob/main/' +
      lab.path
    : 'https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/' +
      lab.path
const lessonUrl = (lesson) => withBase((english ? '/en/' : '/') + lesson)
const sourceUrl = (lab) =>
  repository +
  (/\.(py|md)$/.test(lab.source) ? 'blob/' : 'tree/') +
  'main/' +
  lab.source
let observer

function updatePosition() {
  const el = viewport.value
  if (!el?.clientWidth) return
  const rect = el.getBoundingClientRect()
  const children = [...el.querySelectorAll('.nl-card')]
  const fullyVisible = children
    .map((child, i) => ({ i, rect: child.getBoundingClientRect() }))
    .filter(
      (item) =>
        item.rect.left >= rect.left - 3 && item.rect.right <= rect.right + 3
    )
  start.value =
    fullyVisible[0]?.i ??
    Math.round(
      el.scrollLeft /
        ((children[0]?.getBoundingClientRect().width || 1) + cardGap())
    )
  end.value = fullyVisible.at(-1)?.i ?? start.value
  canPrev.value = el.scrollLeft > 2
  canNext.value = el.scrollLeft + el.clientWidth < el.scrollWidth - 2
}
function cardGap() {
  return (
    Number.parseFloat(
      window.getComputedStyle(viewport.value.querySelector('.nl-track')).gap
    ) || 0
  )
}
function slide(direction) {
  const width =
    viewport.value?.querySelector('.nl-card')?.getBoundingClientRect().width ||
    300
  const gap = cardGap()
  viewport.value?.scrollBy({
    left: direction * (width + gap),
    behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches
      ? 'instant'
      : 'smooth'
  })
}
async function syncViewport() {
  await nextTick()
  observer?.disconnect()
  if (viewport.value) {
    viewport.value.scrollLeft = 0
    observer = new ResizeObserver(updatePosition)
    observer.observe(viewport.value)
    updatePosition()
  }
}
function setChapter(id) {
  selectedChapter.value = id
  if (chapterMenu.value) chapterMenu.value.open = false
  const url = new URLSearchParams(window.location.search)
  url.delete('chapter')
  if (id === 'all') url.delete('section')
  else url.set('section', id)
  window.history.replaceState(
    null,
    '',
    window.location.pathname +
      (url.size ? '?' + url.toString() : '') +
      window.location.hash
  )
  window.dispatchEvent(new Event('lab:context-change'))
}
async function fitChapterMenu() {
  if (!chapterMenu.value?.open) return
  const summary = chapterMenu.value
    .querySelector('summary')
    .getBoundingClientRect()
  const below = window.innerHeight - summary.bottom - 24
  const above = summary.top - 24
  menuAbove.value = below < 200 && above > below
  menuHeight.value = Math.max(
    80,
    Math.min(420, menuAbove.value ? above : below)
  )
  await nextTick()
  chapterMenu.value
    ?.querySelector('[aria-pressed="true"]')
    ?.scrollIntoView({ block: 'nearest' })
}
function clickAway(event) {
  if (chapterMenu.value && !chapterMenu.value.contains(event.target))
    chapterMenu.value.open = false
}
async function showDetails(lab) {
  active.value = lab
  await nextTick()
  modal.value.showModal()
}
function closeDetails() {
  modal.value?.close()
}
function backdropClick(event) {
  if (event.target !== modal.value) return
  const r = modal.value.getBoundingClientRect()
  if (
    event.clientX < r.left ||
    event.clientX > r.right ||
    event.clientY < r.top ||
    event.clientY > r.bottom
  )
    closeDetails()
}
watch([selectedChapter, view], syncViewport)
watch(view, (value) => {
  const url = new URL(window.location.href)
  if (value === 'list') url.searchParams.set('view', 'list')
  else url.searchParams.delete('view')
  window.history.replaceState(null, '', url.pathname + url.search + url.hash)
  window.dispatchEvent(new Event('lab:context-change'))
})
onMounted(() => {
  const query = new URLSearchParams(window.location.search)
  const section = resolveLabSection(
    query.get('section') || query.get('chapter')
  )
  if (section !== 'all') setChapter(section)
  else if (query.has('chapter') || query.has('section')) setChapter('all')
  if (new URLSearchParams(window.location.search).get('view') === 'list')
    view.value = 'list'
  syncViewport()
  window.addEventListener('click', clickAway)
  window.addEventListener('resize', fitChapterMenu)
})
onBeforeUnmount(() => {
  observer?.disconnect()
  window.removeEventListener('click', clickAway)
  window.removeEventListener('resize', fitChapterMenu)
})
</script>

<template>
  <div class="notebook-lab">
    <div class="nl-container">
      <header class="nl-heading">
        <div class="nl-eyebrow">
          <FlaskConical :size="15" :stroke-width="1.7" />
          HANDS-ON MODERN RL
        </div>
        <h1>
          {{ copy.heading }}
          <span class="nl-heading-dot">.</span>
        </h1>
        <p>{{ copy.intro }}</p>
        <p class="nl-heading-sub">
          {{ copy.subintro }}
        </p>
      </header>

      <div class="nl-toolbar">
        <div class="nl-filter-area">
          <details
            ref="chapterMenu"
            class="nl-chapter-menu"
            @toggle="fitChapterMenu"
            @keydown.esc="chapterMenu.open = false"
          >
            <summary>
              <BookOpen :size="16" />
              <span>{{ chapterLabel }}</span>
              <ChevronDown :size="14" />
            </summary>
            <div
              class="nl-menu-options"
              :class="{ 'nl-menu-above': menuAbove }"
              :style="{ maxHeight: menuHeight + 'px' }"
              :aria-label="copy.chapterFilter"
            >
              <button
                type="button"
                :aria-pressed="selectedChapter === 'all'"
                @click="setChapter('all')"
              >
                <span>{{ copy.all }}</span>
                <Check v-if="selectedChapter === 'all'" :size="15" />
              </button>
              <button
                v-for="chapter in availableChapters"
                :key="chapter.id"
                type="button"
                :aria-pressed="selectedChapter === chapter.id"
                @click="setChapter(chapter.id)"
              >
                <span>{{ chapter.name }}</span>
                <Check v-if="selectedChapter === chapter.id" :size="15" />
              </button>
            </div>
          </details>
          <span class="nl-total" aria-live="polite">
            {{ filtered.length }}
            {{ english && filtered.length === 1 ? 'Notebook' : copy.notebooks }}
          </span>
        </div>
        <div class="nl-view-switch" :aria-label="copy.browse">
          <button
            type="button"
            :aria-pressed="view === 'cards'"
            :aria-label="copy.cardLabel"
            @click="view = 'cards'"
          >
            <Grid2X2 :size="16" />
            <span>{{ copy.cards }}</span>
          </button>
          <button
            type="button"
            :aria-pressed="view === 'list'"
            :aria-label="copy.listLabel"
            @click="view = 'list'"
          >
            <List :size="17" />
            <span>{{ copy.list }}</span>
          </button>
        </div>
      </div>

      <template v-if="view === 'cards'">
        <div
          ref="viewport"
          class="nl-viewport"
          :aria-label="copy.shelfLabel"
          @scroll.passive="updatePosition"
        >
          <div class="nl-track">
            <article
              v-for="lab in filtered"
              :key="lab.id"
              class="nl-card"
              :data-lab="lab.id"
            >
              <button
                type="button"
                class="nl-cover"
                :class="[
                  'nl-scene-' + lab.id,
                  { 'nl-result-preview': lab.previewKind === 'result' }
                ]"
                :aria-label="
                  copy.review + ' ' + lab.title + ' · ' + copy.details
                "
                @click="showDetails(lab)"
              >
                <img
                  v-if="lab.image"
                  :src="lab.image"
                  :alt="
                    lab.title +
                    ' · ' +
                    (lab.previewKind === 'result'
                      ? copy.experimentPreview
                      : copy.environment)
                  "
                  :loading="lab.id === 'cartpole' ? 'eager' : 'lazy'"
                  decoding="async"
                />
                <div
                  v-else-if="lab.id === 'minigrid'"
                  class="nl-maze"
                  aria-hidden="true"
                >
                  <span
                    v-for="n in 35"
                    :key="n"
                    :class="{
                      wall: [
                        2, 3, 7, 9, 14, 16, 17, 19, 21, 23, 26, 30, 31, 33
                      ].includes(n),
                      agent: n === 29,
                      goal: n === 6
                    }"
                  ></span>
                </div>
                <div
                  v-else-if="lab.id === 'selfplay'"
                  class="nl-board"
                  aria-hidden="true"
                >
                  <span
                    v-for="(mark, i) in [
                      '×',
                      '',
                      '○',
                      '',
                      '×',
                      '',
                      '○',
                      '',
                      '×'
                    ]"
                    :key="i"
                  >
                    {{ mark }}
                  </span>
                </div>
                <div
                  v-else-if="lab.id === 'jax'"
                  class="nl-pixels"
                  aria-hidden="true"
                >
                  <span
                    v-for="n in 80"
                    :key="n"
                    :class="{
                      lit: [
                        4, 6, 13, 14, 15, 16, 17, 22, 23, 25, 27, 28, 32, 33,
                        34, 35, 36, 37, 38, 43, 47, 52, 58, 69, 70, 71
                      ].includes(n)
                    }"
                  ></span>
                </div>
                <div v-else class="nl-agents" aria-hidden="true">
                  <div class="nl-agent a"></div>
                  <div class="nl-agent b"></div>
                  <div class="nl-agent c"></div>
                  <div class="nl-target"></div>
                </div>
                <span class="nl-cover-label">
                  {{
                    lab.previewKind === 'result'
                      ? copy.experimentPreview
                      : lab.image
                        ? copy.environment
                        : copy.schematic
                  }}
                </span>
                <span class="nl-cover-more"><ArrowUpRight :size="17" /></span>
              </button>
              <div class="nl-card-content">
                <div class="nl-card-meta">
                  <span>{{ lab.method }}</span>
                  <span class="nl-meta-separator">·</span>
                  <Cpu :size="12" />
                  <span>{{ lab.device.split(' · ')[0] }}</span>
                </div>
                <button
                  type="button"
                  class="nl-card-title"
                  @click="showDetails(lab)"
                >
                  <h2>{{ lab.title }}</h2>
                  <span>{{ lab.subtitle }}</span>
                </button>
                <p class="nl-card-desc">{{ lab.desc }}</p>
                <div class="nl-card-actions">
                  <a
                    class="nl-open-primary"
                    :href="platformUrl(lab, 'modelscope')"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    ModelScope
                    <ArrowUpRight :size="14" />
                  </a>
                  <a
                    class="nl-open-secondary"
                    :href="platformUrl(lab, 'colab')"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Colab
                    <ArrowUpRight :size="14" />
                  </a>
                </div>
                <a
                  v-if="lab.resultUrl"
                  class="nl-chapter-link nl-result-link"
                  :href="lab.resultUrl"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {{ copy.savedResult }}
                  <ArrowUpRight :size="12" />
                </a>
                <a class="nl-chapter-link" :href="lessonUrl(lab.lesson)">
                  <BookOpen :size="12" />
                  {{ lab.chapterName }}
                  <ArrowRight :size="12" />
                </a>
              </div>
            </article>
          </div>
        </div>
        <div class="nl-shelf-footer">
          <span aria-live="polite">
            {{ start + 1 }}–{{ end + 1 }}
            <span class="nl-range-divider">/</span>
            {{ filtered.length }}
          </span>
          <div class="nl-shelf-controls">
            <span class="nl-swipe-hint">{{ copy.swipe }}</span>
            <button
              type="button"
              :aria-label="copy.previous"
              :disabled="!canPrev"
              @click="slide(-1)"
            >
              <ArrowLeft :size="18" />
            </button>
            <button
              type="button"
              :aria-label="copy.next"
              :disabled="!canNext"
              @click="slide(1)"
            >
              <ArrowRight :size="18" />
            </button>
          </div>
        </div>
      </template>

      <div v-else class="nl-directory">
        <section
          v-for="chapter in visibleChapters"
          :key="chapter.id"
          class="nl-directory-section"
        >
          <h2>{{ chapter.name }}</h2>
          <div
            v-for="lab in filtered.filter(
              (item) => item.chapterId === chapter.id
            )"
            :key="lab.id"
            class="nl-directory-row"
          >
            <button
              class="nl-directory-title"
              type="button"
              @click="showDetails(lab)"
            >
              <span>{{ lab.title }}</span>
              <span>{{ lab.subtitle }}</span>
            </button>
            <span class="nl-directory-device">
              {{ lab.device.split(' · ')[0] }}
            </span>
            <div class="nl-directory-platforms">
              <a
                v-if="lab.resultUrl"
                :href="lab.resultUrl"
                target="_blank"
                rel="noopener noreferrer"
              >
                {{ copy.savedResult }}
                <ArrowUpRight :size="13" />
              </a>
              <a
                :href="platformUrl(lab, 'modelscope')"
                target="_blank"
                rel="noopener noreferrer"
              >
                ModelScope
                <ArrowUpRight :size="13" />
              </a>
              <a
                :href="platformUrl(lab, 'colab')"
                target="_blank"
                rel="noopener noreferrer"
              >
                Colab
                <ArrowUpRight :size="13" />
              </a>
            </div>
          </div>
        </section>
      </div>

      <section class="nl-below">
        <div class="nl-usage">
          <NotebookPen :size="18" :stroke-width="1.6" />
          <div>
            <h2>{{ copy.usage }}</h2>
            <p>
              {{ copy.usageDesc }}
            </p>
          </div>
          <a :href="lessonUrl('chapter01_cartpole/training')">
            {{ copy.start }}
            <ArrowRight :size="14" />
          </a>
        </div>
        <details v-if="planned.length" class="nl-planned">
          <summary>
            <span>{{ copy.planned }}</span>
            <span class="nl-planned-count">
              {{ planned.length }} {{ copy.preparing }}
              <ChevronDown :size="14" />
            </span>
          </summary>
          <div class="nl-planned-grid">
            <a
              v-for="lab in planned"
              :key="lab.id"
              :href="sourceUrl(lab)"
              target="_blank"
              rel="noopener noreferrer"
            >
              <span>{{ lab.name }}</span>
              <span>
                {{ lab.chapterName }}
                <ArrowUpRight :size="13" />
              </span>
            </a>
          </div>
        </details>
        <p class="nl-footnote">{{ copy.footnote }}</p>
      </section>
    </div>

    <dialog
      ref="modal"
      class="nl-dialog"
      aria-labelledby="nl-detail-title"
      @click="backdropClick"
    >
      <template v-if="active">
        <div class="nl-dialog-heading">
          <span>
            <FlaskConical :size="15" />
            {{ copy.details }}
          </span>
          <button
            type="button"
            :aria-label="copy.close"
            autofocus
            @click="closeDetails"
          >
            <X :size="19" />
          </button>
        </div>
        <div v-if="active.image" class="nl-dialog-image">
          <img
            :src="active.image"
            :alt="
              active.title +
              ' · ' +
              (active.previewKind === 'result'
                ? copy.experimentPreview
                : copy.environment)
            "
          />
          <span>
            {{
              active.previewKind === 'result'
                ? copy.experimentPreview
                : copy.bookPreview
            }}
          </span>
        </div>
        <div class="nl-dialog-content">
          <span class="nl-dialog-chapter">{{ active.chapterName }}</span>
          <h2 id="nl-detail-title">{{ active.title }}</h2>
          <p class="nl-dialog-subtitle">{{ active.subtitle }}</p>
          <p class="nl-dialog-description">{{ active.desc }}</p>
          <div class="nl-detail-facts">
            <div>
              <span>{{ copy.method }}</span>
              <strong>{{ active.method }}</strong>
            </div>
            <div>
              <span>{{ copy.resource }}</span>
              <strong>{{ active.device }}</strong>
            </div>
            <div>
              <span>{{ copy.kind }}</span>
              <strong>{{ active.kind }}</strong>
            </div>
          </div>
          <div class="nl-result">
            <h3>{{ copy.result }}</h3>
            <p>{{ active.result }}</p>
          </div>
          <div class="nl-dialog-actions">
            <a
              class="nl-open-primary"
              :href="platformUrl(active, 'modelscope')"
              target="_blank"
              rel="noopener noreferrer"
            >
              {{ copy.openModelscope }}
              <ArrowUpRight :size="15" />
            </a>
            <a
              class="nl-open-secondary"
              :href="platformUrl(active, 'colab')"
              target="_blank"
              rel="noopener noreferrer"
            >
              {{ copy.openColab }}
              <ArrowUpRight :size="15" />
            </a>
          </div>
          <div class="nl-dialog-links">
            <a
              v-if="active.resultUrl"
              :href="active.resultUrl"
              target="_blank"
              rel="noopener noreferrer"
            >
              {{ copy.savedResult }}
              <ArrowUpRight :size="14" />
            </a>
            <a :href="lessonUrl(active.lesson)" @click="closeDetails">
              <BookOpen :size="14" />
              {{ copy.chapter }}
            </a>
            <a
              :href="repository + 'blob/main/' + active.path"
              target="_blank"
              rel="noopener noreferrer"
            >
              {{ copy.source }}
              <ArrowUpRight :size="13" />
            </a>
          </div>
          <details class="nl-runtime">
            <summary>
              {{ copy.runtime }}
              <ChevronDown :size="13" />
            </summary>
            <p>
              {{
                active.previewKind === 'result'
                  ? copy.runtimeExecutedDesc
                  : copy.runtimeDesc
              }}
            </p>
            <p>{{ active.next }}</p>
            <a
              v-if="active.validation === 'local-cpu-executed'"
              :href="
                repository +
                'blob/main/code/chapter-notebooks/' +
                (english ? 'validation.en.json' : 'validation.json')
              "
              target="_blank"
              rel="noopener noreferrer"
            >
              {{ copy.verification }}
              <ArrowUpRight :size="13" />
            </a>
          </details>
        </div>
      </template>
    </dialog>
  </div>
</template>

<style scoped>
.notebook-lab {
  --nl-bg: var(--surface-primary);
  --nl-muted-bg: var(--surface-secondary);
  --nl-line: var(--vp-c-divider);
  --nl-text: var(--vp-c-text-1);
  --nl-muted: var(--vp-c-text-2);
  --nl-brand: var(--vp-c-brand-1);
  background: var(--nl-bg);
  color: var(--nl-text);
  min-height: calc(100vh - 150px);
  padding: 58px 0 42px;
}
.nl-container {
  width: min(1180px, calc(100% - 96px));
  margin: 0 auto;
}
.nl-heading {
  margin-bottom: 42px;
}
.nl-eyebrow {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 10px;
  letter-spacing: 1.8px;
  font-weight: 600;
  color: var(--nl-brand);
  margin-bottom: 16px;
}
.nl-heading h1 {
  font-size: clamp(36px, 4vw, 48px);
  font-weight: 650;
  letter-spacing: -2px;
  line-height: 1.18;
  margin: 0 0 19px;
}
.nl-heading-dot {
  color: var(--nl-brand);
}
.nl-heading p {
  font-size: 17px;
  line-height: 1.65;
  margin: 0;
  letter-spacing: -0.2px;
}
.nl-heading .nl-heading-sub {
  margin-top: 5px;
  font-size: 14px;
  color: var(--nl-muted);
}
.nl-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 22px;
}
.nl-filter-area {
  display: flex;
  align-items: center;
  gap: 18px;
  min-width: 0;
}
.nl-total {
  color: var(--nl-muted);
  font-size: 12px;
  white-space: nowrap;
}
.nl-chapter-menu {
  position: relative;
  z-index: 12;
}
.nl-chapter-menu summary {
  display: flex;
  align-items: center;
  gap: 9px;
  list-style: none;
  border: 1px solid var(--nl-line);
  padding: 9px 12px;
  border-radius: 8px;
  font-size: 13px;
  user-select: none;
  cursor: pointer;
  background: var(--nl-bg);
}
.nl-chapter-menu summary::-webkit-details-marker {
  display: none;
}
.nl-chapter-menu summary > svg:last-child {
  color: var(--nl-muted);
  margin-left: 10px;
  transition: transform 0.18s;
}
.nl-chapter-menu[open] summary > svg:last-child {
  transform: rotate(180deg);
}
.nl-menu-options {
  position: absolute;
  top: calc(100% + 8px);
  left: 0;
  width: 284px;
  padding: 6px;
  border: 1px solid var(--nl-line);
  border-radius: 12px;
  background: var(--nl-bg);
  box-shadow: 0 12px 40px rgb(0 0 0 / 0.1);
  overflow-y: auto;
  overscroll-behavior: contain;
}
.nl-menu-options.nl-menu-above {
  top: auto;
  bottom: calc(100% + 8px);
}
.nl-menu-options button {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 12px;
  border-radius: 6px;
  text-align: left;
  font-size: 13px;
  cursor: pointer;
}
.nl-menu-options button:hover {
  background: var(--nl-muted-bg);
}
.nl-menu-options button[aria-pressed='true'] {
  color: var(--nl-brand);
  background: var(--vp-c-brand-soft);
}
.nl-view-switch {
  display: flex;
  align-items: center;
  padding: 3px;
  border-radius: 8px;
  background: var(--nl-muted-bg);
  gap: 3px;
}
.nl-view-switch button {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  font-size: 12px;
  border-radius: 6px;
  color: var(--nl-muted);
  cursor: pointer;
}
.nl-view-switch button[aria-pressed='true'] {
  background: var(--nl-bg);
  color: var(--nl-text);
  box-shadow: 0 1px 3px rgb(0 0 0 / 0.08);
}
.nl-viewport {
  overflow-x: auto;
  scrollbar-width: none;
  scroll-snap-type: x mandatory;
  overscroll-behavior-x: contain;
  padding: 2px 1px 10px;
  margin: 0 -1px;
}
.nl-viewport::-webkit-scrollbar {
  display: none;
}
.nl-track {
  display: flex;
  gap: 24px;
}
.nl-card {
  flex: 0 0 calc((100% - 48px) / 3);
  min-width: 0;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--nl-line);
  border-radius: 16px;
  background: var(--nl-bg);
  overflow: hidden;
  scroll-snap-align: start;
  transition: box-shadow 0.2s;
}
.nl-card:hover {
  box-shadow: 0 6px 24px rgb(0 0 0 / 0.045);
}
.nl-cover {
  position: relative;
  width: 100%;
  aspect-ratio: 1.68;
  display: grid;
  place-items: center;
  background: var(--nl-muted-bg);
  overflow: hidden;
  cursor: pointer;
  border-bottom: 1px solid var(--nl-line);
}
.nl-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.35s;
}
.nl-cover:hover img {
  transform: scale(1.035);
}
.nl-result-preview img {
  object-fit: contain;
  padding: 12px;
  background: #fff;
}
.nl-scene-cartpole img {
  object-fit: contain;
  padding: 24px 5px 10px;
  background: #fff;
}
.nl-scene-gymnasium img {
  object-fit: contain;
  background: #fff;
}
.nl-scene-atari img {
  object-fit: contain;
  background: #090b14;
  padding: 7px;
}
.nl-cover-label {
  position: absolute;
  bottom: 12px;
  left: 14px;
  background: rgb(255 255 255 / 0.92);
  color: #464a53;
  padding: 2px 7px;
  border: 1px solid rgb(0 0 0 / 0.04);
  border-radius: 4px;
  font-size: 10px;
  line-height: 18px;
}
.nl-cover-more {
  position: absolute;
  right: 12px;
  bottom: 12px;
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  color: #303646;
  background: rgb(255 255 255 / 0.92);
}
.nl-card-content {
  padding: 24px 24px 18px;
  display: flex;
  flex-direction: column;
  flex: 1;
}
.nl-card-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  color: var(--nl-muted);
  font-size: 11px;
  line-height: 18px;
  margin-bottom: 12px;
}
.nl-meta-separator {
  opacity: 0.55;
  padding: 0 2px;
}
.nl-card-title {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  text-align: left;
  cursor: pointer;
}
.nl-card-title h2 {
  font-size: 23px;
  font-weight: 600;
  line-height: 1.3;
  letter-spacing: -0.65px;
  margin: 0 0 7px;
}
.nl-card-title > span {
  font-size: 14px;
  font-weight: 500;
  line-height: 1.5;
}
.nl-card-title:hover h2 {
  color: var(--nl-brand);
}
.nl-card-desc {
  color: var(--nl-muted);
  font-size: 12px;
  line-height: 1.85;
  margin: 12px 0 24px;
  min-height: 45px;
}
.nl-card-actions {
  display: flex;
  gap: 8px;
  margin-top: auto;
}
.nl-open-primary,
.nl-open-secondary {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 10px 13px;
  border-radius: 7px;
  font-size: 12px;
  font-weight: 500;
  line-height: 1.3;
  text-decoration: none;
  min-height: 38px;
  transition: background 0.18s;
}
.nl-open-primary {
  background: var(--vp-c-brand-1);
  color: #fff;
}
.nl-open-primary:hover {
  background: var(--vp-c-brand-2);
}
.nl-open-secondary {
  border: 1px solid var(--nl-line);
  color: var(--nl-text);
}
.nl-open-secondary:hover {
  background: var(--nl-muted-bg);
}
.nl-card-actions .nl-open-primary {
  flex: 1.2;
}
.nl-card-actions .nl-open-secondary {
  flex: 1;
}
.nl-chapter-link {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 10px;
  color: var(--nl-muted);
  padding-top: 18px;
  margin-top: 17px;
  border-top: 1px solid var(--nl-line);
  min-height: 32px;
}
.nl-chapter-link > svg:last-child {
  margin-left: auto;
}
.nl-chapter-link:hover {
  color: var(--nl-brand);
}
.nl-shelf-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 16px;
  color: var(--nl-muted);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}
.nl-range-divider {
  opacity: 0.45;
  padding: 0 9px;
}
.nl-shelf-controls {
  display: flex;
  align-items: center;
  gap: 9px;
}
.nl-swipe-hint {
  margin-right: 12px;
  font-size: 11px;
  color: var(--nl-muted);
}
.nl-shelf-controls button {
  width: 36px;
  height: 36px;
  border: 1px solid var(--nl-line);
  border-radius: 50%;
  display: grid;
  place-items: center;
  cursor: pointer;
  color: var(--nl-text);
}
.nl-shelf-controls button:hover:enabled {
  background: var(--nl-muted-bg);
}
.nl-shelf-controls button:disabled {
  opacity: 0.3;
  cursor: default;
}
.nl-below {
  margin-top: 52px;
}
.nl-usage {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 26px 0;
  border-top: 1px solid var(--nl-line);
  border-bottom: 1px solid var(--nl-line);
}
.nl-usage > svg {
  color: var(--nl-brand);
  flex-shrink: 0;
  margin-top: 4px;
}
.nl-usage h2 {
  font-size: 14px;
  font-weight: 550;
  margin: 0 0 5px;
}
.nl-usage p {
  font-size: 12px;
  color: var(--nl-muted);
  line-height: 1.7;
  margin: 0;
}
.nl-usage > a {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 7px;
  white-space: nowrap;
  font-size: 12px;
  color: var(--nl-brand);
  align-self: center;
}
.nl-planned {
  padding: 23px 0;
  border-bottom: 1px solid var(--nl-line);
}
.nl-planned summary {
  list-style: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  cursor: pointer;
}
.nl-planned summary::-webkit-details-marker {
  display: none;
}
.nl-planned-count {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 11px;
  color: var(--nl-muted);
}
.nl-planned[open] .nl-planned-count svg {
  transform: rotate(180deg);
}
.nl-planned-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0 40px;
  margin-top: 20px;
}
.nl-planned-grid a {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 15px 0;
  border-top: 1px solid var(--nl-line);
  font-size: 13px;
}
.nl-planned-grid a > span:last-child {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: var(--nl-muted);
  font-size: 11px;
}
.nl-planned-grid a:hover {
  color: var(--nl-brand);
}
.nl-footnote {
  margin: 20px 0 0;
  color: var(--nl-muted);
  font-size: 10px;
  line-height: 1.8;
}
.nl-directory-section {
  margin: 0 0 30px;
}
.nl-directory-section > h2 {
  margin: 0;
  padding: 15px 0;
  border-bottom: 1px solid var(--nl-line);
  font-size: 12px;
  color: var(--nl-muted);
  font-weight: 500;
}
.nl-directory-row {
  display: grid;
  grid-template-columns: 1fr 100px 210px;
  align-items: center;
  gap: 20px;
  padding: 19px 0;
  border-bottom: 1px solid var(--nl-line);
}
.nl-directory-title {
  display: flex;
  align-items: baseline;
  gap: 15px;
  text-align: left;
  cursor: pointer;
}
.nl-directory-title > span:first-child {
  font-size: 15px;
  font-weight: 550;
}
.nl-directory-title > span:last-child {
  font-size: 12px;
  color: var(--nl-muted);
}
.nl-directory-title:hover > span:first-child {
  color: var(--nl-brand);
}
.nl-directory-device {
  font-size: 11px;
  color: var(--nl-muted);
}
.nl-directory-platforms {
  display: flex;
  gap: 22px;
  justify-content: flex-end;
}
.nl-directory-platforms a {
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--nl-brand);
  font-size: 12px;
}
.nl-dialog {
  width: min(620px, calc(100% - 32px));
  max-height: calc(100dvh - 48px);
  margin: auto;
  padding: 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 18px;
  background: var(--surface-primary);
  color: var(--vp-c-text-1);
  box-shadow: 0 24px 100px rgb(0 0 0 / 0.2);
  overscroll-behavior: contain;
}
.nl-dialog::backdrop {
  background: rgb(17 24 39 / 0.42);
  backdrop-filter: blur(5px);
}
.nl-dialog-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 21px;
  border-bottom: 1px solid var(--nl-line);
}
.nl-dialog-heading > span {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12px;
  color: var(--nl-muted);
}
.nl-dialog-heading > button {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  cursor: pointer;
}
.nl-dialog-heading > button:hover {
  background: var(--nl-muted-bg);
}
.nl-dialog-image {
  height: 210px;
  background: #fff;
  display: grid;
  place-items: center;
  position: relative;
  border-bottom: 1px solid var(--nl-line);
}
.nl-dialog-image img {
  max-height: 100%;
  width: 100%;
  object-fit: contain;
}
.nl-dialog-image > span {
  position: absolute;
  left: 18px;
  bottom: 12px;
  font-size: 10px;
  padding: 2px 7px;
  background: rgb(255 255 255 / 0.92);
  color: #464a53;
  border-radius: 4px;
}
.nl-dialog-content {
  padding: 28px 32px 24px;
}
.nl-dialog-chapter {
  font-size: 11px;
  color: var(--nl-brand);
}
.nl-dialog-content h2 {
  font-size: 30px;
  line-height: 1.3;
  font-weight: 600;
  letter-spacing: -0.7px;
  margin: 10px 0 4px;
}
.nl-dialog-subtitle {
  font-size: 15px;
  margin: 0 0 16px;
}
.nl-dialog-description {
  font-size: 13px;
  line-height: 1.8;
  color: var(--nl-muted);
  margin: 0;
}
.nl-detail-facts {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  padding: 22px 0;
  margin: 20px 0;
  border-top: 1px solid var(--nl-line);
  border-bottom: 1px solid var(--nl-line);
}
.nl-detail-facts > div {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.nl-detail-facts span {
  color: var(--nl-muted);
  font-size: 10px;
}
.nl-detail-facts strong {
  font-size: 12px;
  font-weight: 500;
}
.nl-result h3 {
  font-size: 12px;
  font-weight: 550;
  margin: 0 0 6px;
}
.nl-result p {
  font-size: 12px;
  line-height: 1.8;
  color: var(--nl-muted);
  margin: 0;
}
.nl-dialog-actions {
  display: flex;
  gap: 10px;
  margin: 24px 0 18px;
}
.nl-dialog-actions > a {
  flex: 1;
}
.nl-dialog-links {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
}
.nl-dialog-links a {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--nl-brand);
}
.nl-runtime {
  font-size: 11px;
  color: var(--nl-muted);
  border-top: 1px solid var(--nl-line);
  margin-top: 23px;
  padding-top: 16px;
}
.nl-runtime summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  list-style: none;
  cursor: pointer;
}
.nl-runtime p {
  margin: 12px 0 0;
  line-height: 1.8;
}
.nl-maze {
  display: grid;
  grid-template-columns: repeat(7, 24px);
  gap: 3px;
  transform: rotate(-8deg);
}
.nl-maze span {
  width: 24px;
  height: 24px;
  border-radius: 2px;
  background: #dce2f0;
}
.nl-maze span.wall {
  background: #67748d;
}
.nl-maze span.agent {
  background: #5363c2;
  box-shadow: inset 0 0 0 5px #adb8f6;
}
.nl-maze span.goal {
  background: #f6c66f;
}
.nl-board {
  display: grid;
  grid-template-columns: repeat(3, 42px);
  gap: 2px;
  background: #bfc9e1;
}
.nl-board span {
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  background: #f1f4fa;
  color: #687ac0;
  font-size: 34px;
  font-weight: 400;
}
.nl-pixels {
  display: grid;
  grid-template-columns: repeat(10, 11px);
  gap: 3px;
}
.nl-pixels span {
  width: 11px;
  height: 11px;
}
.nl-pixels span.lit {
  background: #818ce4;
  box-shadow: 0 0 10px rgb(129 140 228 / 0.25);
}
.nl-scene-jax {
  background: #1d243c;
}
.nl-agents {
  width: 160px;
  height: 130px;
  position: relative;
  border-bottom: 1px solid #d3daeb;
  background: radial-gradient(circle at 50% 50%, #d5ddf0 1px, transparent 1px);
  background-size: 16px 16px;
}
.nl-agent {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  position: absolute;
  background: #7e8bd1;
  border: 5px solid #c5cef0;
}
.nl-agent.a {
  left: 10px;
  top: 16px;
}
.nl-agent.b {
  left: 66px;
  top: 80px;
}
.nl-agent.c {
  right: 0;
  top: 25px;
}
.nl-target {
  width: 14px;
  height: 14px;
  background: #e9b45b;
  border-radius: 3px;
  position: absolute;
  left: 77px;
  top: 37px;
  transform: rotate(45deg);
}
@media (min-width: 1500px) {
  .notebook-lab {
    padding-top: 70px;
  }
}
@media (max-width: 1100px) {
  .nl-container {
    width: calc(100% - 64px);
  }
  .nl-card-content {
    padding: 21px 20px 17px;
  }
  .nl-card-title h2 {
    font-size: 21px;
  }
  .nl-open-primary,
  .nl-open-secondary {
    padding: 10px 8px;
  }
  .nl-directory-title {
    flex-direction: column;
    gap: 3px;
  }
}
@media (max-width: 850px) {
  .nl-card {
    flex-basis: calc((100% - 24px) / 2);
  }
  .nl-heading {
    margin-bottom: 32px;
  }
  .notebook-lab {
    padding-top: 38px;
  }
  .nl-usage {
    flex-wrap: wrap;
  }
  .nl-usage > a {
    margin-left: 32px;
  }
  .nl-directory-row {
    grid-template-columns: 1fr 65px 190px;
    gap: 10px;
  }
}
@media (max-width: 560px) {
  .nl-container {
    width: calc(100% - 40px);
  }
  .notebook-lab {
    padding-top: 32px;
  }
  .nl-heading h1 {
    font-size: 38px;
  }
  .nl-heading p {
    font-size: 15px;
  }
  .nl-heading .nl-heading-sub {
    font-size: 12px;
  }
  .nl-eyebrow {
    font-size: 9px;
  }
  .nl-card {
    flex-basis: calc(100% - 20px);
  }
  .nl-track {
    gap: 16px;
  }
  .nl-card-content {
    padding: 22px 22px 17px;
  }
  .nl-filter-area {
    gap: 10px;
  }
  .nl-total {
    display: none;
  }
  .nl-view-switch button {
    padding: 7px 8px;
  }
  .nl-view-switch button span {
    display: none;
  }
  .nl-chapter-menu summary {
    max-width: 230px;
    gap: 7px;
    padding: 9px 10px;
    font-size: 11px;
  }
  .nl-chapter-menu summary > svg:last-child {
    margin-left: 3px;
  }
  .nl-menu-options {
    width: min(284px, calc(100vw - 40px));
  }
  .nl-swipe-hint {
    display: none;
  }
  .nl-below {
    margin-top: 32px;
  }
  .nl-planned-grid {
    grid-template-columns: 1fr;
  }
  .nl-planned summary {
    font-size: 12px;
  }
  .nl-planned-count {
    font-size: 10px;
    gap: 5px;
  }
  .nl-directory-row {
    grid-template-columns: 1fr auto;
  }
  .nl-directory-device {
    text-align: right;
  }
  .nl-directory-platforms {
    grid-column: 1/-1;
    justify-content: flex-start;
    padding-top: 4px;
    gap: 24px;
  }
  .nl-dialog-content {
    padding: 24px 22px;
  }
  .nl-dialog-image {
    height: 160px;
  }
  .nl-dialog-actions {
    flex-direction: column;
  }
  .nl-dialog-content h2 {
    font-size: 28px;
  }
  .nl-detail-facts {
    gap: 9px;
  }
  .nl-detail-facts strong {
    font-size: 11px;
  }
  .nl-footnote {
    font-size: 10px;
  }
}
@media (pointer: coarse) {
  .nl-shelf-controls button,
  .nl-dialog-heading > button {
    width: 44px;
    height: 44px;
  }
  .nl-open-primary,
  .nl-open-secondary,
  .nl-chapter-menu summary,
  .nl-view-switch button,
  .nl-menu-options button {
    min-height: 44px;
  }
  .nl-directory-platforms a {
    min-height: 40px;
  }
  .nl-chapter-link {
    min-height: 44px;
  }
}
@media (prefers-reduced-motion: reduce) {
  .nl-cover img,
  .nl-card,
  .nl-chapter-menu summary > svg:last-child {
    transition: none;
  }
}
</style>
