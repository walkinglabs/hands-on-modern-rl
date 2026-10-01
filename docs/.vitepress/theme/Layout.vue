<script setup>
import DefaultTheme from 'vitepress/theme'
import { useData, useRoute, useRouter, withBase } from 'vitepress'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  PopoverContent,
  PopoverPortal,
  PopoverRoot,
  PopoverTrigger
} from 'reka-ui'
import {
  ChevronsDown,
  ChevronsUp,
  HandHeart,
  Maximize,
  MessageCircle,
  Minimize,
  Moon,
  Settings,
  Sun
} from 'lucide-vue-next'
import ReadingProgress from './components/ReadingProgress.vue'
import SidebarFooter from './components/SidebarFooter.vue'
import TextType from './components/TextType.vue'
import mediumZoom from 'medium-zoom'
import { initGithubStars } from './githubStars.js'

const { frontmatter, site, theme, isDark, page } = useData()
const route = useRoute()
const router = useRouter()

const FONT_SIZE_STORAGE_KEY = 'ct-doc-font-size'
const LINE_HEIGHT_STORAGE_KEY = 'ct-doc-line-height'
const DOC_WIDTH_STORAGE_KEY = 'ct-doc-content-width'
const SIDEBAR_COLLAPSED_KEY = 'ct-sidebar-collapsed'
const SIDEBAR_WIDTH_KEY = 'ct-sidebar-width-reading-v5'
const DISCORD_URL = 'https://discord.gg/XU7DQmpqk'

const MIN_FONT_SIZE = 15
const MAX_FONT_SIZE = 20
const DEFAULT_FONT_SIZE = 16
const MIN_LINE_HEIGHT = 1.55
const MAX_LINE_HEIGHT = 2
const DEFAULT_LINE_HEIGHT = 1.75
const MIN_DOC_WIDTH = 780
const MAX_DOC_WIDTH = 1280
const DEFAULT_DOC_WIDTH = 980

const DEFAULT_SIDEBAR_WIDTH = 280
const MIN_SIDEBAR_WIDTH = 264
const MAX_SIDEBAR_WIDTH = 520
const ROUTE_LOADING_DELAY_MS = 120
const ROUTE_LOADING_TIMEOUT_MS = 5000
const ROUTE_RECOVERY_MAX_ATTEMPTS = 5
const ROUTE_RECOVERY_BASE_DELAY_MS = 400

const SIDEBAR_NAV_GROUP_PREFIXES = {
  '/preface/intro': '序章 · 导论',
  '/chapter01_cartpole/principles': 'Part I ·',
  '/chapter07_dqn/from-q-to-dqn': 'Part II ·',
  '/chapter15_rlhf/base-model-to-assistant': 'Part IV ·',
  '/chapter22_agentic/overview': 'Part V ·',
  '/chapter30_alignment_failures/classical-failures': 'Part VII ·',
  '/appendix_industrial_training/training-debugging': '附录',
  '/en/preface/intro': 'Preface ·',
  '/en/chapter01_cartpole/principles': 'Fundamentals & Classical RL',
  '/en/chapter07_dqn/from-q-to-dqn': 'Deep Reinforcement Learning',
  '/en/chapter15_rlhf/base-model-to-assistant': 'LLM Alignment & Post-Training',
  '/en/chapter22_agentic/intro': 'Agentic Reinforcement Learning',
  '/en/chapter32_selfplay/self-play-outlook':
    'Safety, Evaluation & Research Frontiers',
  '/en/appendix_industrial_training/intro': 'Appendices'
}

const LEGACY_INTRO_REDIRECTS = {
  '/chapter01_cartpole/intro': '/chapter01_cartpole/principles',
  '/chapter02_bandits/intro': '/chapter03_mdp/bandit',
  '/chapter03_mdp/intro': '/chapter03_mdp/bandit',
  '/chapter07_dqn/intro': '/chapter07_dqn/from-q-to-dqn',
  '/chapter08_policy_gradient/intro':
    '/chapter08_policy_gradient/policy-gradient',
  '/chapter09_actor_critic/intro': '/chapter09_actor_critic/advantage-function',
  '/chapter15_rlhf/intro': '/chapter15_rlhf/base-model-to-assistant',
  '/chapter21_cai_rlvr/intro': '/chapter21_cai_rlvr/hhh-practice',
  '/chapter21_cai_rlvr/rlaif-engineering': '/chapter21_cai_rlvr/hhh-practice',
  '/chapter20_prm_search/intro': '/chapter20_prm_search/outcome-vs-process',
  '/chapter22_agentic/intro': '/chapter22_agentic/overview',
  '/chapter23_rl_based_swe/intro': '/chapter23_rl_based_swe/swe-bench-and-rlvr',
  '/chapter24_deep_research/intro':
    '/chapter24_deep_research/browser-rl-harness',
  '/chapter25_computer_use/intro': '/chapter25_computer_use/training',
  '/chapter26_vlm/intro': '/chapter26_vlm/vlm-challenges',
  '/chapter27_audio_rl/intro': '/chapter27_audio_rl/reward-design',
  '/chapter30_alignment_failures/intro':
    '/chapter30_alignment_failures/classical-failures',
  '/chapter31_alphaevolve/intro': '/chapter32_selfplay/alphaevolve/',
  '/chapter32_selfplay/intro': '/chapter32_selfplay/self-play-outlook',
  '/chapter13_imitation_meta_rl/intro':
    '/chapter13_imitation_meta_rl/bc-dagger',
  '/appendix_common_pitfalls/intro':
    '/appendix_industrial_training/training-debugging',
  '/appendix_code_cheatsheet/intro': '/appendix_code_cheatsheet/sft-kl',
  '/appendix_game_projects/intro': '/appendix_paper_reading/intro',
  '/appendix_industrial_training/intro':
    '/appendix_industrial_training/training-debugging',
  '/appendix_math/intro': '/appendix_math/linear-algebra-basics',
  '/en/chapter01_cartpole/intro': '/en/chapter01_cartpole/principles',
  '/en/chapter03_mdp/intro': '/en/chapter03_mdp/bandit',
  '/en/chapter07_dqn/intro': '/en/chapter07_dqn/from-q-to-dqn',
  '/en/chapter08_policy_gradient/intro':
    '/en/chapter08_policy_gradient/policy-gradient',
  '/en/chapter09_actor_critic/intro':
    '/en/chapter09_actor_critic/advantage-function',
  '/en/chapter15_rlhf/intro': '/en/chapter15_rlhf/base-model-to-assistant',
  '/en/chapter26_vlm/intro': '/en/chapter26_vlm/vlm-challenges',
  '/en/chapter32_selfplay/intro': '/en/chapter32_selfplay/self-play-outlook',
  '/en/appendix_common_pitfalls/intro':
    '/en/appendix_industrial_training/intro',
  '/en/appendix_code_cheatsheet/intro': '/en/appendix_code_cheatsheet/sft-kl',
  '/en/appendix_math/intro': '/en/appendix_math/linear-algebra-basics'
}

const fontSize = ref(DEFAULT_FONT_SIZE)
const lineHeight = ref(DEFAULT_LINE_HEIGHT)
const docWidth = ref(DEFAULT_DOC_WIDTH)
const readingToolsOpen = ref(false)
const supportOpen = ref(false)
const supportQrWide = ref(true)
const sidebarCollapsed = ref(false)
const sidebarWidth = ref(DEFAULT_SIDEBAR_WIDTH)
const sidebarResizing = ref(false)
const allSidebarGroupsExpanded = ref(false)
const routeLoading = ref(false)
const fullscreenReading = ref(false)

const mermaidViewerOpen = ref(false)
const mermaidViewerSrc = ref('')
const mermaidViewerAlt = ref('')
const mermaidViewerScale = ref(1)
const mermaidViewerScroll = ref(null)
const mermaidViewerNaturalWidth = ref(0)
const mermaidViewerNaturalHeight = ref(0)
const mermaidViewerCustomZoom = ref(false)
const mermaidViewerDragging = ref(false)

const isHomePage = computed(() => frontmatter.value.layout === 'home')
const showDocChrome = computed(() => !isHomePage.value)
const mobileRoutePath = computed(() => {
  const base = site.value.base || '/'
  if (base === '/') return route.path

  const normalizedBase = base.replace(/\/$/, '')
  const pathWithoutBase = route.path.replace(normalizedBase, '')
  return pathWithoutBase || '/'
})
const isEnglishRoute = computed(
  () =>
    mobileRoutePath.value === '/en' || mobileRoutePath.value.startsWith('/en/')
)
const sidebarGroupsToggleLabel = computed(() =>
  allSidebarGroupsExpanded.value ? 'Collapse all' : 'Expand all'
)
const currentLanguage = computed(() =>
  isEnglishRoute.value ? 'English' : '简体中文'
)
const alternateLanguage = computed(() =>
  isEnglishRoute.value ? '简体中文' : 'English'
)
const alternateLanguageLink = computed(() => {
  if (isEnglishRoute.value) {
    const zhPath = mobileRoutePath.value.replace(/^\/en(?=\/|$)/, '') || '/'
    return withBase(zhPath)
  }

  const enPath =
    mobileRoutePath.value === '/' ? '/en/' : `/en${mobileRoutePath.value}`
  const englishRoutes = theme.value.englishRoutes || []
  return withBase(englishRoutes.includes(enPath) ? enPath : '/en/')
})
const alternateLanguageShort = computed(() =>
  isEnglishRoute.value ? '中' : 'EN'
)
const alternateLanguageAriaLabel = computed(() =>
  isEnglishRoute.value ? '切换到简体中文' : 'Switch to English'
)
const languageToggleAriaCurrent = computed(() => {
  const isOnAlternateRoot =
    alternateLanguageLink.value === withBase('/') ||
    alternateLanguageLink.value === withBase('/en/') ||
    alternateLanguageLink.value === withBase('/en')
  return isOnAlternateRoot ? null : 'true'
})
const supportButtonLabel = computed(() =>
  isEnglishRoute.value ? 'Give the creator a like' : '给制作者一个赞吧'
)
const settingsButtonLabel = computed(() =>
  isEnglishRoute.value ? 'Reading and appearance settings' : '阅读与外观设置'
)
const fullscreenReadingLabel = computed(() =>
  isEnglishRoute.value
    ? fullscreenReading.value
      ? 'Exit fullscreen reading'
      : 'Enter fullscreen reading'
    : fullscreenReading.value
      ? '退出全屏阅读'
      : '进入全屏阅读'
)
const supportNote = computed(() =>
  isEnglishRoute.value
    ? 'Thanks for following this project. Sharing it or joining the discussion is welcome; your attention is the greatest support.'
    : '感谢关注项目，欢迎帮忙宣传或者一起交流，你的关注就是最大的支持。'
)
const supportQrLabel = computed(() =>
  isEnglishRoute.value
    ? 'Official account / community QR code'
    : '公众号 / 社群二维码'
)
const supportQrZoomLabel = computed(() =>
  isEnglishRoute.value ? 'Open WeChat QR code image' : '放大查看微信群二维码'
)
const discordLinkLabel = computed(() =>
  isEnglishRoute.value ? 'Join Discord' : '加入 Discord'
)
const discordLinkMeta = computed(() =>
  isEnglishRoute.value ? 'Community' : '社区交流'
)
const readingToolsCopy = computed(() =>
  isEnglishRoute.value
    ? {
        appearance: 'Appearance',
        light: 'Light',
        dark: 'Dark',
        fontSize: 'Font size',
        lineHeight: 'Line height',
        docWidth: 'Content width',
        decreaseFont: 'A-',
        increaseFont: 'A+',
        default: 'Default',
        narrower: 'Narrower',
        wider: 'Wider',
        tighter: 'Tighter',
        looser: 'Looser',
        switchLight: 'Switch to light mode',
        switchDark: 'Switch to dark mode'
      }
    : {
        appearance: '外观',
        light: '浅色',
        dark: '深色',
        fontSize: '字号',
        lineHeight: '行距',
        docWidth: '正文宽度',
        decreaseFont: 'A-',
        increaseFont: 'A+',
        default: '默认',
        narrower: '更窄',
        wider: '更宽',
        tighter: '更紧',
        looser: '更松',
        switchLight: '切换到浅色模式',
        switchDark: '切换到深色模式'
      }
)
const fontSizeLabel = computed(() => `${clampFontSize(fontSize.value)}px`)
const lineHeightLabel = computed(() =>
  clampLineHeight(lineHeight.value).toFixed(2)
)
const homeTypingText = computed(
  () =>
    frontmatter.value.hero?.typingTagline ||
    frontmatter.value.hero?.tagline ||
    ''
)

let sidebarResizeLeft = 0
let outlineObserver = null
let sidebarObserver = null
let navigationSyncTimer = null
let routeLoadingTimer = null
let routeLoadingFallbackTimer = null
let routeRecoveryTimer = null
let routeRecoveryPath = ''
let routeRecoveryAttempts = 0
let zoom = null
let mermaidViewerDragState = null

const MERMAID_VIEWER_MIN_SCALE = 0.02
const MERMAID_VIEWER_MAX_SCALE = 6
const MERMAID_VIEWER_SCALE_STEP = 0.25

const mermaidViewerScaleLabel = computed(
  () => `${Math.round(mermaidViewerScale.value * 100)}%`
)

const mermaidViewerImageStyle = computed(() => {
  if (!mermaidViewerNaturalWidth.value) {
    return {
      maxHeight: '100%',
      maxWidth: '100%'
    }
  }

  return {
    height: `${Math.max(
      1,
      mermaidViewerNaturalHeight.value * mermaidViewerScale.value
    )}px`,
    width: `${Math.max(1, mermaidViewerNaturalWidth.value * mermaidViewerScale.value)}px`
  }
})

const mermaidViewerStageStyle = computed(() => {
  if (!mermaidViewerNaturalWidth.value) return {}

  return {
    height: `${Math.max(
      1,
      mermaidViewerNaturalHeight.value * mermaidViewerScale.value
    )}px`,
    width: `${Math.max(1, mermaidViewerNaturalWidth.value * mermaidViewerScale.value)}px`
  }
})

const homeTaglineTyping = {
  typingSpeed: 42,
  deletingSpeed: 20,
  pauseDuration: 2600,
  initialDelay: 120
}

function clampFontSize(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return DEFAULT_FONT_SIZE
  return Math.min(MAX_FONT_SIZE, Math.max(MIN_FONT_SIZE, numeric))
}

function clampLineHeight(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return DEFAULT_LINE_HEIGHT
  return Math.min(MAX_LINE_HEIGHT, Math.max(MIN_LINE_HEIGHT, numeric))
}

function clampDocWidth(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return DEFAULT_DOC_WIDTH
  return Math.min(MAX_DOC_WIDTH, Math.max(MIN_DOC_WIDTH, numeric))
}

function clampSidebarWidth(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return DEFAULT_SIDEBAR_WIDTH
  return Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, numeric))
}

function applyFontSize(size) {
  if (typeof document === 'undefined') return
  document.documentElement.style.setProperty('--ct-doc-font-size', `${size}px`)
}

function applyLineHeight(value) {
  if (typeof document === 'undefined') return
  document.documentElement.style.setProperty(
    '--ct-doc-line-height',
    String(value)
  )
}

function applyDocWidth(width) {
  if (typeof document === 'undefined') return
  document.documentElement.style.setProperty(
    '--vp-doc-content-max-width',
    `${width}px`
  )
}

function applySidebarWidth(width) {
  if (typeof document === 'undefined') return
  document.documentElement.style.setProperty('--vp-sidebar-width', `${width}px`)
}

function updateSidebarEdgePosition() {
  if (typeof document === 'undefined') return
  const sidebar = document.querySelector('.VPSidebar')
  if (!sidebar) return
  document.documentElement.style.setProperty(
    '--ct-sidebar-edge-right',
    `${sidebar.getBoundingClientRect().right}px`
  )
}

function getSidebarScrollContainer() {
  if (typeof document === 'undefined') return null
  return document.querySelector('.VPSidebar > .nav')
}

function setSidebarWidth(value, persist = true) {
  const normalized = clampSidebarWidth(value)
  sidebarWidth.value = normalized
  applySidebarWidth(normalized)
  window.requestAnimationFrame(updateSidebarEdgePosition)

  if (persist && typeof localStorage !== 'undefined') {
    localStorage.setItem(SIDEBAR_WIDTH_KEY, String(normalized))
  }
}

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

function getSidebarGroups() {
  if (typeof document === 'undefined') return []
  return Array.from(document.querySelectorAll('.VPSidebarItem.collapsible'))
}

function syncSidebarGroupsExpanded() {
  const groups = getSidebarGroups()
  allSidebarGroupsExpanded.value =
    groups.length > 0 &&
    groups.every((groupEl) => !groupEl.classList.contains('collapsed'))
}

/**
 * 展开或折叠侧边栏里的全部可折叠分组。
 * VitePress 每个分组用 .VPSidebarItem.collapsible 渲染，collapsed 状态
 * 反映在 CSS class 上；点击组头的 .caret 即可切换。深层分组始终在 DOM 里，
 * 所以一次遍历就能处理所有层级。
 */
function setAllSidebarGroups(expand) {
  const groups = getSidebarGroups()

  // 展开时先处理父级，收起时先处理子级，避免隐藏节点的点击冒泡
  // 反复切换父级，导致“全部收起”最终又回到展开状态。
  const orderedGroups = expand ? groups : groups.reverse()

  orderedGroups.forEach((groupEl) => {
    const isCollapsed = groupEl.classList.contains('collapsed')
    if ((expand && isCollapsed) || (!expand && !isCollapsed)) {
      const toggleEl = groupEl.querySelector(':scope > .item > .caret')
      toggleEl && toggleEl.click()
    }
  })

  syncSidebarGroupsExpanded()
}

function toggleAllSidebarGroups() {
  const shouldExpand = !allSidebarGroupsExpanded.value
  setAllSidebarGroups(shouldExpand)

  // 长目录收起后高度会骤减。归位到顶部，避免首个分组被吸顶工具栏遮住。
  if (!shouldExpand) {
    requestAnimationFrame(() => {
      getSidebarScrollContainer()?.scrollTo({ top: 0 })
    })
  }
}

function closeReadingTools() {
  readingToolsOpen.value = false
}

function closeSupportPanel() {
  supportOpen.value = false
}

function syncFullscreenReading() {
  fullscreenReading.value = Boolean(document.fullscreenElement)
}

async function toggleFullscreenReading() {
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen()
    } else {
      await document.documentElement.requestFullscreen()
    }
  } catch {
    // 部分移动浏览器不支持 Fullscreen API，保留当前阅读界面。
  }
}

function setAppearance(dark) {
  isDark.value = dark
}

function showRouteLoading() {
  window.clearTimeout(routeLoadingTimer)
  window.clearTimeout(routeLoadingFallbackTimer)
  routeLoading.value = false

  routeLoadingTimer = window.setTimeout(() => {
    routeLoading.value = true
  }, ROUTE_LOADING_DELAY_MS)

  routeLoadingFallbackTimer = window.setTimeout(() => {
    routeLoading.value = false
  }, ROUTE_LOADING_TIMEOUT_MS)
}

function updateSupportQrRatio(event) {
  const image = event.currentTarget
  supportQrWide.value =
    image.naturalWidth > 0 && image.naturalWidth > image.naturalHeight * 1.15
}

function openSupportQrViewer(event) {
  const image = event.currentTarget.querySelector('img')
  if (!image) return
  openMermaidViewer(image)
}

function hideRouteLoading() {
  window.clearTimeout(routeLoadingTimer)
  window.clearTimeout(routeLoadingFallbackTimer)
  routeLoadingTimer = null
  routeLoadingFallbackTimer = null
  routeLoading.value = false
}

function normalizeNavigationPath(path) {
  if (!path) return '/'
  const withoutHash = path.split(/[?#]/, 1)[0]
  const withoutHtml = withoutHash.replace(/\.html$/, '')
  return withoutHtml === '/' ? '/' : withoutHtml.replace(/\/$/, '')
}

function redirectLegacyIntroRoute() {
  const target =
    LEGACY_INTRO_REDIRECTS[normalizeNavigationPath(mobileRoutePath.value)]
  if (!target) return false

  void router.go(withBase(target))
  return true
}

function isPrimaryNavigationRoute(path) {
  const normalizedPath = normalizeNavigationPath(path)
  return (theme.value.nav || []).some(
    (item) => normalizeNavigationPath(item.link) === normalizedPath
  )
}

function clearRouteRecovery() {
  window.clearTimeout(routeRecoveryTimer)
  routeRecoveryTimer = null
  routeRecoveryPath = ''
  routeRecoveryAttempts = 0
}

function scheduleRouteRecovery() {
  const path = normalizeNavigationPath(mobileRoutePath.value)

  if (!page.value.isNotFound || !isPrimaryNavigationRoute(path)) {
    clearRouteRecovery()
    return
  }

  if (routeRecoveryPath !== path) {
    window.clearTimeout(routeRecoveryTimer)
    routeRecoveryTimer = null
    routeRecoveryPath = path
    routeRecoveryAttempts = 0
  }

  if (
    routeRecoveryTimer ||
    routeRecoveryAttempts >= ROUTE_RECOVERY_MAX_ATTEMPTS
  ) {
    return
  }

  const delay = ROUTE_RECOVERY_BASE_DELAY_MS * 2 ** routeRecoveryAttempts
  routeRecoveryTimer = window.setTimeout(async () => {
    routeRecoveryTimer = null
    routeRecoveryAttempts += 1
    await router.go(window.location.href)
  }, delay)
}

function clampMermaidViewerScale(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return 1
  return Math.min(
    MERMAID_VIEWER_MAX_SCALE,
    Math.max(MERMAID_VIEWER_MIN_SCALE, numeric)
  )
}

function openMermaidViewer(image) {
  if (!image) return
  mermaidViewerSrc.value = image.currentSrc || image.src
  mermaidViewerAlt.value = image.alt || 'Mermaid diagram'
  mermaidViewerNaturalWidth.value = image.naturalWidth || 0
  mermaidViewerNaturalHeight.value = image.naturalHeight || 0
  mermaidViewerScale.value = 1
  mermaidViewerCustomZoom.value = false
  mermaidViewerOpen.value = true
  document.body.classList.add('ct-mermaid-viewer-open')
  nextTick(() => {
    window.requestAnimationFrame(fitMermaidViewerToScreen)
  })
}

function closeMermaidViewer() {
  mermaidViewerOpen.value = false
  mermaidViewerSrc.value = ''
  mermaidViewerAlt.value = ''
  mermaidViewerNaturalWidth.value = 0
  mermaidViewerNaturalHeight.value = 0
  mermaidViewerCustomZoom.value = false
  mermaidViewerDragging.value = false
  mermaidViewerDragState = null
  document.body.classList.remove('ct-mermaid-viewer-open')
}

function centerMermaidViewer() {
  const scroll = mermaidViewerScroll.value
  if (!scroll) return
  scroll.scrollLeft = Math.max(0, (scroll.scrollWidth - scroll.clientWidth) / 2)
  scroll.scrollTop = Math.max(
    0,
    (scroll.scrollHeight - scroll.clientHeight) / 2
  )
}

function getMermaidViewerFitScale() {
  const scroll = mermaidViewerScroll.value
  const width = mermaidViewerNaturalWidth.value
  const height = mermaidViewerNaturalHeight.value
  if (!scroll || !width || !height) return 1

  const styles = window.getComputedStyle(scroll)
  const availableWidth =
    scroll.clientWidth -
    Number.parseFloat(styles.paddingLeft || 0) -
    Number.parseFloat(styles.paddingRight || 0)
  const availableHeight =
    scroll.clientHeight -
    Number.parseFloat(styles.paddingTop || 0) -
    Number.parseFloat(styles.paddingBottom || 0)

  return clampMermaidViewerScale(
    Math.min(availableWidth / width, availableHeight / height, 1)
  )
}

function fitMermaidViewerToScreen() {
  mermaidViewerScale.value = getMermaidViewerFitScale()
  nextTick(() => {
    window.requestAnimationFrame(centerMermaidViewer)
  })
}

function setMermaidViewerScale(value, anchorEvent = null) {
  const scroll = mermaidViewerScroll.value
  const previousScale = mermaidViewerScale.value
  const nextScale = clampMermaidViewerScale(value)
  if (Math.abs(nextScale - previousScale) < 0.001) return

  let anchor = null
  if (scroll && anchorEvent) {
    const rect = scroll.getBoundingClientRect()
    anchor = {
      offsetX: anchorEvent.clientX - rect.left,
      offsetY: anchorEvent.clientY - rect.top,
      scrollX: scroll.scrollLeft,
      scrollY: scroll.scrollTop,
      ratio: nextScale / previousScale
    }
  }

  mermaidViewerCustomZoom.value = true
  mermaidViewerScale.value = nextScale

  nextTick(() => {
    window.requestAnimationFrame(() => {
      if (!scroll || !anchor) {
        centerMermaidViewer()
        return
      }

      scroll.scrollLeft =
        (anchor.scrollX + anchor.offsetX) * anchor.ratio - anchor.offsetX
      scroll.scrollTop =
        (anchor.scrollY + anchor.offsetY) * anchor.ratio - anchor.offsetY
    })
  })
}

function zoomMermaidViewer(delta) {
  setMermaidViewerScale(mermaidViewerScale.value + delta)
}

function resetMermaidViewerZoom() {
  mermaidViewerCustomZoom.value = false
  fitMermaidViewerToScreen()
}

function handleMermaidViewerImageLoad(event) {
  const image = event.currentTarget
  mermaidViewerNaturalWidth.value = image.naturalWidth || 0
  mermaidViewerNaturalHeight.value = image.naturalHeight || 0
  if (!mermaidViewerCustomZoom.value) {
    fitMermaidViewerToScreen()
  }
}

function handleMermaidViewerWheel(event) {
  if (!mermaidViewerOpen.value) return
  event.preventDefault()

  const delta = event.deltaY || event.deltaX
  if (!delta) return

  const factor = Math.exp(-delta * 0.0015)
  setMermaidViewerScale(mermaidViewerScale.value * factor, event)
}

function handleMermaidViewerPointerDown(event) {
  if (!mermaidViewerOpen.value || event.button !== 0) return
  const scroll = mermaidViewerScroll.value
  if (!scroll) return

  mermaidViewerDragState = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    scrollLeft: scroll.scrollLeft,
    scrollTop: scroll.scrollTop
  }
  mermaidViewerDragging.value = true
  event.currentTarget.setPointerCapture?.(event.pointerId)
  event.preventDefault()
}

function handleMermaidViewerPointerMove(event) {
  const scroll = mermaidViewerScroll.value
  const drag = mermaidViewerDragState
  if (!scroll || !drag || drag.pointerId !== event.pointerId) return

  scroll.scrollLeft = drag.scrollLeft - (event.clientX - drag.startX)
  scroll.scrollTop = drag.scrollTop - (event.clientY - drag.startY)
}

function stopMermaidViewerDrag(event = null) {
  if (event && mermaidViewerDragState?.pointerId !== event.pointerId) return
  mermaidViewerDragging.value = false
  mermaidViewerDragState = null
}

function handleMermaidViewerKeydown(event) {
  if (!mermaidViewerOpen.value) return

  if (event.key === 'Escape') {
    event.preventDefault()
    closeMermaidViewer()
    return
  }

  if (event.key === '+' || event.key === '=') {
    event.preventDefault()
    zoomMermaidViewer(MERMAID_VIEWER_SCALE_STEP)
    return
  }

  if (event.key === '-' || event.key === '_') {
    event.preventDefault()
    zoomMermaidViewer(-MERMAID_VIEWER_SCALE_STEP)
    return
  }

  if (event.key === '0') {
    event.preventDefault()
    resetMermaidViewerZoom()
  }
}

function handleMermaidImageClick(event) {
  const image = event.target.closest('img[data-mermaid-viewer="true"]')
  if (!image) return
  event.preventDefault()
  event.stopPropagation()
  openMermaidViewer(image)
}

function initMermaidViewer() {
  if (typeof document === 'undefined') return
  document.querySelectorAll('.main').forEach((main) => {
    main.removeEventListener('click', handleMermaidImageClick)
    main.addEventListener('click', handleMermaidImageClick)
  })
}

function cleanupMermaidViewer() {
  if (typeof document === 'undefined') return
  document.querySelectorAll('.main').forEach((main) => {
    main.removeEventListener('click', handleMermaidImageClick)
  })
}

function resetFontSize() {
  setFontSize(DEFAULT_FONT_SIZE)
}

function resetLineHeight() {
  setLineHeight(DEFAULT_LINE_HEIGHT)
}

function setFontSize(value) {
  fontSize.value = clampFontSize(value)
}

function setLineHeight(value) {
  lineHeight.value = clampLineHeight(value)
}

function resetDocWidth() {
  docWidth.value = DEFAULT_DOC_WIDTH
}

function decreaseFontSize() {
  setFontSize(fontSize.value - 1)
}

function increaseFontSize() {
  setFontSize(fontSize.value + 1)
}

function decreaseLineHeight() {
  setLineHeight(lineHeight.value - 0.05)
}

function increaseLineHeight() {
  setLineHeight(lineHeight.value + 0.05)
}

function updateFontSizeFromRange(event) {
  setFontSize(event.currentTarget.valueAsNumber)
}

function updateLineHeightFromRange(event) {
  setLineHeight(event.currentTarget.valueAsNumber)
}

function narrowDocWidth() {
  docWidth.value = clampDocWidth(docWidth.value - 40)
}

function widenDocWidth() {
  docWidth.value = clampDocWidth(docWidth.value + 40)
}

function getSidebarLeftBoundary() {
  const sidebar = document.querySelector('.VPSidebar')
  if (!sidebar) return 0
  return sidebar.getBoundingClientRect().left
}

function updateSidebarWidthFromPointer(clientX) {
  const nextWidth = clientX - sidebarResizeLeft
  setSidebarWidth(nextWidth, false)
}

function handleSidebarResizeMove(event) {
  if (!sidebarResizing.value) return
  updateSidebarWidthFromPointer(event.clientX)
}

function stopSidebarResize() {
  if (!sidebarResizing.value) return

  sidebarResizing.value = false
  document.body.classList.remove('ct-sidebar-resizing')
  localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth.value))
  window.removeEventListener('pointermove', handleSidebarResizeMove)
  window.removeEventListener('pointerup', stopSidebarResize)
  window.removeEventListener('pointercancel', stopSidebarResize)
}

function startSidebarResize(event) {
  if (typeof window === 'undefined') return
  if (window.innerWidth < 960 || sidebarCollapsed.value) return

  event.preventDefault()
  sidebarResizeLeft = getSidebarLeftBoundary()
  sidebarResizing.value = true
  document.body.classList.add('ct-sidebar-resizing')
  updateSidebarWidthFromPointer(event.clientX)

  window.addEventListener('pointermove', handleSidebarResizeMove)
  window.addEventListener('pointerup', stopSidebarResize)
  window.addEventListener('pointercancel', stopSidebarResize)
}

function handleViewportResize() {
  setSidebarWidth(sidebarWidth.value, false)
  updateSidebarEdgePosition()
  if (mermaidViewerOpen.value && !mermaidViewerCustomZoom.value) {
    window.requestAnimationFrame(fitMermaidViewerToScreen)
  }
}

function handleWindowKeydown(event) {
  handleMermaidViewerKeydown(event)
  if (event.defaultPrevented) return

  if (event.key === 'Escape') {
    closeReadingTools()
    closeSupportPanel()
  }
}

function scrollOutlineToActiveItem(activeLink) {
  const outlineContainer = document.querySelector('.VPDocAsideOutline')
  if (!outlineContainer || !activeLink) return

  const containerRect = outlineContainer.getBoundingClientRect()
  const linkRect = activeLink.getBoundingClientRect()
  const linkTop = linkRect.top - containerRect.top + outlineContainer.scrollTop
  const targetScrollTop =
    linkTop - containerRect.height / 2 + linkRect.height / 2

  const isAbove = linkRect.top < containerRect.top + 20
  const isBelow = linkRect.bottom > containerRect.bottom - 20

  if (isAbove || isBelow) {
    outlineContainer.scrollTo({
      top: targetScrollTop,
      behavior: 'smooth'
    })
  }
}

function scrollSidebarToActiveItem(activeItem) {
  const sidebarContainer = getSidebarScrollContainer()
  if (!sidebarContainer || !activeItem) return

  const target =
    activeItem.querySelector('.item') ||
    activeItem.querySelector('a') ||
    activeItem

  const containerRect = sidebarContainer.getBoundingClientRect()
  const targetRect = target.getBoundingClientRect()
  const targetTop =
    targetRect.top - containerRect.top + sidebarContainer.scrollTop
  const targetScrollTop =
    targetTop - containerRect.height / 2 + targetRect.height / 2
  const isInside =
    targetRect.top >= containerRect.top + 16 &&
    targetRect.bottom <= containerRect.bottom - 16

  if (!isInside) {
    sidebarContainer.scrollTo({
      top: targetScrollTop,
      behavior: 'smooth'
    })
  }
}

function setActiveSidebarGroup(targetGroup) {
  document
    .querySelectorAll('.VPSidebar > .nav > .group.ct-nav-group-active')
    .forEach((group) => {
      group.classList.remove('ct-nav-group-active')
      group.removeAttribute('aria-current')
    })

  if (!targetGroup) return
  targetGroup.classList.add('ct-nav-group-active')
  targetGroup.setAttribute('aria-current', 'location')
}

function getSidebarNavigationGroup(path) {
  const sidebarContainer = getSidebarScrollContainer()
  const prefix = SIDEBAR_NAV_GROUP_PREFIXES[normalizeNavigationPath(path)]
  if (!sidebarContainer || !prefix) return null

  const targetGroup = Array.from(
    sidebarContainer.querySelectorAll(':scope > .group')
  ).find((group) => {
    const title = group.querySelector(
      '.group-title, .VPSidebarItem.level-0 > .item .text'
    )
    return title?.textContent?.trim().startsWith(prefix)
  })
  if (!targetGroup) return null

  return { sidebarContainer, targetGroup }
}

function scrollSidebarToNavigationGroup(path) {
  const target = getSidebarNavigationGroup(path)
  if (!target) return false

  const { sidebarContainer, targetGroup } = target

  setActiveSidebarGroup(targetGroup)

  const containerRect = sidebarContainer.getBoundingClientRect()
  const groupRect = targetGroup.getBoundingClientRect()
  const targetScrollTop =
    groupRect.top - containerRect.top + sidebarContainer.scrollTop - 8

  sidebarContainer.scrollTo({
    top: Math.max(0, targetScrollTop),
    behavior: 'smooth'
  })
  return true
}

function cleanupNavigationSync() {
  outlineObserver?.disconnect()
  sidebarObserver?.disconnect()
  outlineObserver = null
  sidebarObserver = null

  if (navigationSyncTimer) {
    window.clearTimeout(navigationSyncTimer)
    navigationSyncTimer = null
  }
}

function initMediumZoom() {
  if (typeof document === 'undefined') return
  if (zoom) zoom.detach()
  zoom = mediumZoom('.main img:not([data-mermaid-viewer="true"])', {
    background: 'rgba(15, 23, 42, 0.62)',
    margin: 24
  })
}

function renderSidebarKatex() {
  // no-op: sidebar labels are plain text
}

function renderOutlineMath() {
  if (typeof document === 'undefined') return
  document
    .querySelectorAll('.VPDocAsideOutline .outline-link')
    .forEach((link) => {
      const href = link.getAttribute('href')
      if (!href) return
      const id = decodeURIComponent(href.split('#')[1] || '')
      if (!id) return
      const heading = document.getElementById(id)
      if (!heading) return
      const clone = heading.cloneNode(true)
      clone.querySelector('.header-anchor')?.remove()
      const html = clone.innerHTML.trim()
      if (!html || html === link.innerHTML) return
      link.innerHTML = html
    })
}

function scheduleRenderOutlineMath() {
  if (typeof window === 'undefined') return
  window.requestAnimationFrame(renderOutlineMath)
}

function enhanceNavTitle() {
  if (typeof document === 'undefined') return
  const title = document.querySelector('.VPNavBar .title')
  if (!title) return

  if (title.tagName === 'A') {
    title.href = withBase('/preface/intro')
  }

  const titleText = title.querySelector('span:last-of-type') || title
  if (titleText.dataset.ctEnhancedTitle === 'true') return

  const text = titleText.textContent?.trim()
  if (text !== 'Hands on Modern RL') return

  titleText.dataset.ctEnhancedTitle = 'true'
  titleText.classList.add('ct-nav-title-text')
  titleText.innerHTML =
    '<span class="ct-nav-title-main">Hands on </span><span class="ct-nav-title-accent">Modern RL</span>'
}

function initNavigationSync() {
  cleanupNavigationSync()

  navigationSyncTimer = window.setTimeout(() => {
    const outlineContainer = document.querySelector('.VPDocAsideOutline')
    const sidebarContainer = getSidebarScrollContainer()

    if (outlineContainer) {
      outlineObserver = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
          const target = mutation.target
          if (target.classList?.contains('active') && target.tagName === 'A') {
            scrollOutlineToActiveItem(target)
          }
        }
      })

      outlineObserver.observe(outlineContainer, {
        attributes: true,
        subtree: true,
        attributeFilter: ['class']
      })

      const currentActive = outlineContainer.querySelector('.active')
      if (currentActive) {
        scrollOutlineToActiveItem(currentActive)
      }
    }

    if (sidebarContainer) {
      sidebarObserver = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
          const target = mutation.target
          if (target.classList?.contains('is-active')) {
            scrollSidebarToActiveItem(target)
          }
        }

        if (
          !sidebarContainer.querySelector(':scope > .group.ct-nav-group-active')
        ) {
          const navigationGroup = getSidebarNavigationGroup(
            mobileRoutePath.value
          )
          setActiveSidebarGroup(navigationGroup?.targetGroup)
        }

        syncSidebarGroupsExpanded()
      })

      sidebarObserver.observe(sidebarContainer, {
        attributes: true,
        childList: true,
        subtree: true,
        attributeFilter: ['class']
      })

      const syncedToNavigationGroup = scrollSidebarToNavigationGroup(
        mobileRoutePath.value
      )
      const currentSidebarActive = sidebarContainer.querySelector('.is-active')
      if (!syncedToNavigationGroup && currentSidebarActive) {
        setActiveSidebarGroup(currentSidebarActive.closest('.group'))
        scrollSidebarToActiveItem(currentSidebarActive)
      } else if (!syncedToNavigationGroup) {
        setActiveSidebarGroup(null)
      }

      syncSidebarGroupsExpanded()
    }

    updateSidebarEdgePosition()
  }, 80)
}

onMounted(() => {
  const savedFontSize = clampFontSize(
    localStorage.getItem(FONT_SIZE_STORAGE_KEY)
  )
  const savedLineHeight = clampLineHeight(
    localStorage.getItem(LINE_HEIGHT_STORAGE_KEY)
  )
  const savedDocWidth = clampDocWidth(
    localStorage.getItem(DOC_WIDTH_STORAGE_KEY)
  )
  const savedSidebarWidth = localStorage.getItem(SIDEBAR_WIDTH_KEY)
  const savedCollapsed = localStorage.getItem(SIDEBAR_COLLAPSED_KEY)

  fontSize.value = savedFontSize
  lineHeight.value = savedLineHeight
  docWidth.value = savedDocWidth
  applyFontSize(savedFontSize)
  applyLineHeight(savedLineHeight)
  applyDocWidth(savedDocWidth)

  if (savedSidebarWidth) {
    setSidebarWidth(savedSidebarWidth, false)
  } else {
    setSidebarWidth(DEFAULT_SIDEBAR_WIDTH, false)
  }

  sidebarCollapsed.value = savedCollapsed === 'true'
  document.body.classList.toggle('ct-sidebar-collapsed', sidebarCollapsed.value)

  window.addEventListener('resize', handleViewportResize)
  window.addEventListener('keydown', handleWindowKeydown)
  document.addEventListener('fullscreenchange', syncFullscreenReading)
  syncFullscreenReading()
  initNavigationSync()
  window.requestAnimationFrame(syncSidebarGroupsExpanded)
  updateSidebarEdgePosition()
  initMediumZoom()
  initMermaidViewer()
  enhanceNavTitle()
  renderSidebarKatex()
  scheduleRenderOutlineMath()
  initGithubStars(theme)

  router.onBeforeRouteChange = () => {
    showRouteLoading()
  }

  router.onAfterRouteChange = () => {
    hideRouteLoading()
    if (redirectLegacyIntroRoute()) return
    scheduleRouteRecovery()
  }

  if (!redirectLegacyIntroRoute()) {
    scheduleRouteRecovery()
  }
})

onBeforeUnmount(() => {
  stopSidebarResize()
  cleanupNavigationSync()
  cleanupMermaidViewer()
  closeMermaidViewer()
  hideRouteLoading()
  clearRouteRecovery()
  router.onBeforeRouteChange = undefined
  router.onAfterRouteChange = undefined
  window.removeEventListener('resize', handleViewportResize)
  window.removeEventListener('keydown', handleWindowKeydown)
  document.removeEventListener('fullscreenchange', syncFullscreenReading)
})

watch(fontSize, (next) => {
  const normalized = clampFontSize(next)
  if (fontSize.value !== normalized) {
    fontSize.value = normalized
  }
  applyFontSize(normalized)
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(FONT_SIZE_STORAGE_KEY, String(normalized))
  }
})

watch(
  () => route.path,
  () => {
    hideRouteLoading()
  }
)

watch(lineHeight, (next) => {
  const normalized = clampLineHeight(next)
  if (lineHeight.value !== normalized) {
    lineHeight.value = normalized
  }
  applyLineHeight(normalized)
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(LINE_HEIGHT_STORAGE_KEY, String(normalized))
  }
})

watch(docWidth, (next) => {
  const normalized = clampDocWidth(next)
  applyDocWidth(normalized)
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(DOC_WIDTH_STORAGE_KEY, String(normalized))
  }
})

watch(sidebarCollapsed, (collapsed) => {
  if (typeof document === 'undefined') return
  document.body.classList.toggle('ct-sidebar-collapsed', collapsed)
  localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(collapsed))
  window.requestAnimationFrame(updateSidebarEdgePosition)
})

watch(readingToolsOpen, (open) => {
  if (open) supportOpen.value = false
})

watch(supportOpen, (open) => {
  if (open) readingToolsOpen.value = false
})

watch(
  () => route.path,
  async () => {
    await nextTick()
    initNavigationSync()
    initMediumZoom()
    initMermaidViewer()
    enhanceNavTitle()
    renderSidebarKatex()
    scheduleRenderOutlineMath()
    window.requestAnimationFrame(updateSidebarEdgePosition)
  }
)
</script>

<template>
  <PopoverRoot v-model:open="readingToolsOpen">
    <DefaultTheme.Layout>
      <template v-if="showDocChrome" #nav-bar-content-after>
        <div class="ct-nav-tools">
          <a
            class="ct-nav-tool-button ct-language-toggle"
            :href="alternateLanguageLink"
            :aria-label="alternateLanguageAriaLabel"
            :title="alternateLanguageAriaLabel"
            :aria-current="languageToggleAriaCurrent"
          >
            <span class="ct-language-toggle-text">
              {{ alternateLanguageShort }}
            </span>
          </a>

          <button
            class="ct-nav-tool-button"
            type="button"
            :aria-label="fullscreenReadingLabel"
            :title="fullscreenReadingLabel"
            @click="toggleFullscreenReading"
          >
            <Minimize
              v-if="fullscreenReading"
              :size="18"
              :stroke-width="2"
              aria-hidden="true"
            />
            <Maximize
              v-else
              :size="18"
              :stroke-width="2"
              aria-hidden="true"
            />
          </button>

          <PopoverPortal>
            <Transition name="ct-reading-tools-fade">
              <PopoverContent
                class="ct-popover-content"
                :side-offset="10"
                align="end"
                side="bottom"
              >
                <div class="ct-popover-surface ct-reading-tools-panel">
                  <div class="ct-reading-tools-group">
                    <div class="ct-reading-tools-header">
                      <div class="ct-reading-tools-title">
                        {{ readingToolsCopy.appearance }}
                      </div>
                      <div class="ct-reading-tools-value">
                        {{
                          isDark
                            ? readingToolsCopy.dark
                            : readingToolsCopy.light
                        }}
                      </div>
                    </div>
                    <div
                      class="ct-appearance-toggle"
                      role="group"
                      :aria-label="readingToolsCopy.appearance"
                    >
                      <button
                        class="ct-reading-tools-action"
                        :class="{ active: !isDark }"
                        type="button"
                        :aria-label="readingToolsCopy.switchLight"
                        @click="setAppearance(false)"
                      >
                        <Sun :size="18" :stroke-width="2" aria-hidden="true" />
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        :class="{ active: isDark }"
                        type="button"
                        :aria-label="readingToolsCopy.switchDark"
                        @click="setAppearance(true)"
                      >
                        <Moon :size="18" :stroke-width="2" aria-hidden="true" />
                      </button>
                    </div>
                  </div>

                  <div class="ct-reading-tools-group">
                    <div class="ct-reading-tools-header">
                      <div class="ct-reading-tools-title">
                        {{ readingToolsCopy.fontSize }}
                      </div>
                      <div class="ct-reading-tools-value">
                        {{ fontSizeLabel }}
                      </div>
                    </div>
                    <div class="ct-reading-tools-actions">
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="decreaseFontSize"
                      >
                        {{ readingToolsCopy.decreaseFont }}
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="resetFontSize"
                      >
                        {{ readingToolsCopy.default }}
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="increaseFontSize"
                      >
                        {{ readingToolsCopy.increaseFont }}
                      </button>
                    </div>
                    <input
                      class="ct-reading-tools-range"
                      type="range"
                      :value="fontSize"
                      :min="MIN_FONT_SIZE"
                      :max="MAX_FONT_SIZE"
                      step="1"
                      @input="updateFontSizeFromRange"
                    />
                  </div>

                  <div class="ct-reading-tools-group">
                    <div class="ct-reading-tools-header">
                      <div class="ct-reading-tools-title">
                        {{ readingToolsCopy.lineHeight }}
                      </div>
                      <div class="ct-reading-tools-value">
                        {{ lineHeightLabel }}
                      </div>
                    </div>
                    <div class="ct-reading-tools-actions">
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="decreaseLineHeight"
                      >
                        {{ readingToolsCopy.tighter }}
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="resetLineHeight"
                      >
                        {{ readingToolsCopy.default }}
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="increaseLineHeight"
                      >
                        {{ readingToolsCopy.looser }}
                      </button>
                    </div>
                    <input
                      class="ct-reading-tools-range"
                      type="range"
                      :value="lineHeight"
                      :min="MIN_LINE_HEIGHT"
                      :max="MAX_LINE_HEIGHT"
                      step="0.05"
                      @input="updateLineHeightFromRange"
                    />
                  </div>
                  <div class="ct-reading-tools-group">
                    <div class="ct-reading-tools-header">
                      <div class="ct-reading-tools-title">
                        {{ readingToolsCopy.docWidth }}
                      </div>
                      <div class="ct-reading-tools-value">{{ docWidth }}px</div>
                    </div>
                    <div class="ct-reading-tools-actions">
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="narrowDocWidth"
                      >
                        {{ readingToolsCopy.narrower }}
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="resetDocWidth"
                      >
                        {{ readingToolsCopy.default }}
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="widenDocWidth"
                      >
                        {{ readingToolsCopy.wider }}
                      </button>
                    </div>
                    <input
                      v-model.number="docWidth"
                      class="ct-reading-tools-range"
                      type="range"
                      :min="MIN_DOC_WIDTH"
                      :max="MAX_DOC_WIDTH"
                      step="20"
                    />
                  </div>
                </div>
              </PopoverContent>
            </Transition>
          </PopoverPortal>

          <PopoverRoot v-model:open="supportOpen">
            <PopoverTrigger as-child>
              <button
                class="ct-nav-tool-button"
                type="button"
                :aria-label="supportButtonLabel"
                :title="supportButtonLabel"
              >
                <HandHeart :size="18" :stroke-width="2" aria-hidden="true" />
              </button>
            </PopoverTrigger>

            <PopoverPortal>
              <Transition name="ct-reading-tools-fade">
                <PopoverContent
                  class="ct-popover-content"
                  :side-offset="10"
                  align="end"
                  side="bottom"
                >
                  <div
                    class="ct-popover-surface ct-support-panel"
                    :class="{ 'has-wide-qr': supportQrWide }"
                  >
                    <a
                      class="ct-support-link"
                      href="https://github.com/walkinglabs"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <span>WalkingLab</span>
                      <span>GitHub</span>
                    </a>
                    <a
                      class="ct-support-link"
                      :href="DISCORD_URL"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <span class="ct-support-link-main">
                        <MessageCircle
                          :size="15"
                          :stroke-width="2"
                          aria-hidden="true"
                        />
                        <span>{{ discordLinkLabel }}</span>
                      </span>
                      <span>{{ discordLinkMeta }}</span>
                    </a>
                    <div
                      class="ct-support-qr-card"
                      :class="{ 'is-wide': supportQrWide }"
                    >
                      <button
                        class="ct-support-qr-zoom-button"
                        type="button"
                        :aria-label="supportQrZoomLabel"
                        :title="supportQrZoomLabel"
                        @click="openSupportQrViewer"
                      >
                        <img
                          src="https://github.com/walkinglabs/.github/raw/main/profile/wechat.png"
                          :alt="
                            isEnglishRoute
                              ? 'WalkingLab community QR code'
                              : 'WalkingLab 微信二维码'
                          "
                          loading="lazy"
                          decoding="async"
                          @load="updateSupportQrRatio"
                        />
                      </button>
                      <div>{{ supportQrLabel }}</div>
                    </div>
                    <p class="ct-support-note">
                      {{ supportNote }}
                    </p>
                  </div>
                </PopoverContent>
              </Transition>
            </PopoverPortal>
          </PopoverRoot>
        </div>
      </template>

      <template v-if="showDocChrome" #sidebar-nav-before>
        <Teleport defer to=".VPSidebar">
          <div
            class="ct-sidebar-groups-toolbar"
            role="group"
            :aria-label="isEnglishRoute ? 'Sidebar controls' : '侧边栏分组控制'"
          >
            <div class="ct-sidebar-toolbar-row">
              <div class="ct-sidebar-groups-control">
                <button
                  class="ct-sidebar-groups-button"
                  :class="{ 'is-collapse-action': allSidebarGroupsExpanded }"
                  type="button"
                  :title="sidebarGroupsToggleLabel"
                  :aria-label="sidebarGroupsToggleLabel"
                  :aria-expanded="allSidebarGroupsExpanded"
                  @click="toggleAllSidebarGroups"
                >
                  <ChevronsUp
                    v-if="allSidebarGroupsExpanded"
                    :size="12"
                    :stroke-width="1.8"
                    aria-hidden="true"
                  />
                  <ChevronsDown
                    v-else
                    :size="12"
                    :stroke-width="1.8"
                    aria-hidden="true"
                  />
                  <span>{{ sidebarGroupsToggleLabel }}</span>
                </button>
              </div>
              <SidebarFooter class="ct-sidebar-toolbar-actions">
                <template #settings>
                  <PopoverTrigger as-child>
                    <button
                      class="ct-sidebar-footer-btn"
                      type="button"
                      :title="settingsButtonLabel"
                      :aria-label="settingsButtonLabel"
                    >
                      <Settings
                        :size="16"
                        :stroke-width="2"
                        aria-hidden="true"
                      />
                    </button>
                  </PopoverTrigger>
                </template>
              </SidebarFooter>
            </div>
          </div>
        </Teleport>
      </template>

      <template v-if="isHomePage && homeTypingText" #home-hero-info-after>
        <div class="ct-home-typed-tagline">
          <ClientOnly>
            <TextType :text="homeTypingText" v-bind="homeTaglineTyping" />
          </ClientOnly>
        </div>
      </template>

      <template #nav-screen-content-after>
        <div class="ct-mobile-language-switcher">
          <div class="ct-mobile-language-title">
            {{ isEnglishRoute ? 'Change language' : '切换语言' }}
          </div>
          <div class="ct-mobile-language-options">
            <span class="ct-mobile-language-current">
              {{ currentLanguage }}
            </span>
            <a class="ct-mobile-language-link" :href="alternateLanguageLink">
              {{ alternateLanguage }}
            </a>
          </div>
        </div>
      </template>
    </DefaultTheme.Layout>

    <ClientOnly>
      <div
        v-if="showDocChrome"
        class="ct-sidebar-hover-area"
        :class="{ collapsed: sidebarCollapsed, resizing: sidebarResizing }"
      >
        <div
          v-if="!sidebarCollapsed"
          class="ct-sidebar-resizer"
          role="separator"
          aria-orientation="vertical"
          @pointerdown="startSidebarResize"
        />

        <button
          class="ct-sidebar-toggle-btn"
          :class="{ collapsed: sidebarCollapsed }"
          type="button"
          :aria-label="
            isEnglishRoute
              ? sidebarCollapsed
                ? 'Expand sidebar'
                : 'Collapse sidebar'
              : sidebarCollapsed
                ? '展开目录'
                : '收起目录'
          "
          @click="toggleSidebar"
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path
              v-if="!sidebarCollapsed"
              d="M8 1L3 6l5 5"
              stroke="currentColor"
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="1.5"
            />
            <path
              v-else
              d="M4 1l5 5-5 5"
              stroke="currentColor"
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="1.5"
            />
          </svg>
        </button>
      </div>
    </ClientOnly>

    <ClientOnly>
      <ReadingProgress v-if="showDocChrome" />
    </ClientOnly>

    <ClientOnly>
      <Transition name="ct-route-loading-fade">
        <div
          v-if="routeLoading"
          class="ct-route-loading"
          :aria-label="isEnglishRoute ? 'Loading page' : '页面加载中'"
          aria-live="polite"
        >
          <span class="ct-route-loading-spinner" aria-hidden="true"></span>
        </div>
      </Transition>
    </ClientOnly>

    <ClientOnly>
      <Teleport to="body">
        <div
          v-if="mermaidViewerOpen"
          class="ct-mermaid-viewer"
          role="dialog"
          aria-modal="true"
          :aria-label="isEnglishRoute ? 'View diagram' : '查看图表'"
          @click.self="closeMermaidViewer"
        >
          <div class="ct-mermaid-viewer-toolbar">
            <div class="ct-mermaid-viewer-help">
              <strong>{{ mermaidViewerScaleLabel }}</strong>
              <span>
                {{
                  isEnglishRoute
                    ? 'Wheel to zoom · Drag to pan · + / - to zoom · 0 to reset · Esc to close'
                    : '滚轮缩放 · 拖拽移动 · + / - 缩放 · 0 重置 · Esc 关闭'
                }}
              </span>
            </div>
            <div class="ct-mermaid-viewer-actions">
              <button
                type="button"
                :aria-label="isEnglishRoute ? 'Zoom out' : '缩小图表'"
                @click="zoomMermaidViewer(-MERMAID_VIEWER_SCALE_STEP)"
              >
                -
              </button>
              <button type="button" @click="resetMermaidViewerZoom">
                {{ isEnglishRoute ? 'Reset' : '重置' }}
              </button>
              <button
                type="button"
                :aria-label="isEnglishRoute ? 'Zoom in' : '放大图表'"
                @click="zoomMermaidViewer(MERMAID_VIEWER_SCALE_STEP)"
              >
                +
              </button>
              <button
                type="button"
                :aria-label="isEnglishRoute ? 'Close diagram' : '关闭图表'"
                @click="closeMermaidViewer"
              >
                {{ isEnglishRoute ? 'Close' : '关闭' }}
              </button>
            </div>
          </div>
          <div
            ref="mermaidViewerScroll"
            class="ct-mermaid-viewer-scroll"
            :class="{ 'is-dragging': mermaidViewerDragging }"
            @pointerdown="handleMermaidViewerPointerDown"
            @pointermove="handleMermaidViewerPointerMove"
            @pointerup="stopMermaidViewerDrag"
            @pointercancel="stopMermaidViewerDrag"
            @pointerleave="stopMermaidViewerDrag"
            @wheel="handleMermaidViewerWheel"
          >
            <div
              class="ct-mermaid-viewer-stage"
              :style="mermaidViewerStageStyle"
            >
              <img
                class="ct-mermaid-viewer-image"
                :src="mermaidViewerSrc"
                :alt="mermaidViewerAlt"
                :style="mermaidViewerImageStyle"
                @load="handleMermaidViewerImageLoad"
                @dragstart.prevent
              />
            </div>
          </div>
        </div>
      </Teleport>
    </ClientOnly>
  </PopoverRoot>
</template>

<style>
.ct-nav-tools {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-left: 6px;
}

.VPNavBar .ct-nav-tools {
  order: 21;
}

.VPNavBar .appearance {
  display: none;
}

/* Hide VitePress's default language dropdowns on all viewports.
   Replaced by .ct-language-toggle above, which is a direct link to the
   other locale. VPFlyout's hover-to-open + click-to-toggle combo is
   unreliable on touch devices and on touchscreen laptops, where the
   menu briefly flashes open on mouseenter then immediately closes on
   click (open = !open flips back to false). */
.VPNavBar .VPNavBarTranslations,
.VPNavScreen .VPNavScreenTranslations {
  display: none !important;
}

.ct-language-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 34px;
  height: 34px;
  padding: 0 10px;
  font-size: 12.5px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-decoration: none;
  color: rgba(29, 29, 31, 0.62);
}

.ct-language-toggle:hover,
.ct-language-toggle:focus-visible {
  color: rgba(29, 29, 31, 0.82);
  border-color: rgba(0, 0, 0, 0.05);
  background: rgba(0, 0, 0, 0.04);
}

.ct-nav-tool-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 34px;
  min-width: 34px;
  padding: 0 10px;
  border: 1px solid transparent;
  border-radius: 999px;
  background: transparent;
  color: rgba(29, 29, 31, 0.58);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0;
  box-shadow: none;
  cursor: pointer;
  transition:
    background-color 0.18s ease,
    color 0.2s ease,
    border-color 0.18s ease;
}

.ct-nav-tool-button:hover,
.ct-nav-tool-button:focus-visible,
.ct-nav-tool-button[data-state='open'] {
  border-color: rgba(0, 0, 0, 0.05);
  background: rgba(0, 0, 0, 0.04);
  color: rgba(29, 29, 31, 0.82);
}

.ct-popover-content {
  z-index: 80;
  outline: none;
}

[data-reka-popper-content-wrapper]:has(> .ct-popover-content) {
  z-index: 80 !important;
}

.ct-popover-surface {
  transform: translateY(0) scale(1);
  transform-origin: var(--reka-popover-content-transform-origin, top right);
  will-change: transform;
}

.ct-reading-tools-panel {
  width: 280px;
  padding: 14px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 12px;
  background: #fff;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.1);
}

.ct-reading-tools-group {
  display: grid;
  gap: 10px;
}

.ct-reading-tools-group + .ct-reading-tools-group {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid rgba(148, 163, 184, 0.16);
}

.ct-reading-tools-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.ct-reading-tools-title {
  font-size: 13px;
  font-weight: 700;
  color: var(--vp-c-text-1);
}

.ct-reading-tools-value {
  font-size: 12px;
  color: var(--vp-c-text-2);
}

.ct-reading-tools-actions {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

.ct-reading-tools-action {
  height: 34px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.02);
  color: var(--vp-c-text-1);
  font-size: 13px;
  cursor: pointer;
  transition:
    background-color 0.2s ease,
    border-color 0.2s ease,
    color 0.2s ease;
}

.ct-reading-tools-action:hover {
  border-color: rgba(15, 118, 110, 0.32);
  color: var(--vp-c-brand-1);
}

.ct-reading-tools-action.active {
  border-color: rgba(63, 81, 181, 0.36);
  background: rgba(63, 81, 181, 0.1);
  color: var(--vp-c-brand-1);
  font-weight: 700;
}

.ct-appearance-toggle {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}

.ct-appearance-toggle .ct-reading-tools-action {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
}

.ct-reading-tools-range {
  width: 100%;
  accent-color: var(--vp-c-brand-1);
}

.ct-support-panel {
  width: min(260px, calc(100vw - 24px));
  padding: 12px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 12px;
  background: #fff;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.1);
}

.ct-support-panel.has-wide-qr {
  width: min(520px, calc(100vw - 24px));
}

.ct-support-link {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 38px;
  padding: 0 10px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 8px;
  color: var(--vp-c-text-1);
  font-size: 13px;
  font-weight: 700;
  text-decoration: none;
  transition:
    background-color 0.2s ease,
    border-color 0.2s ease,
    color 0.2s ease;
}

.ct-support-link span:last-child {
  color: var(--vp-c-text-2);
  font-size: 12px;
  font-weight: 500;
}

.ct-support-link:hover {
  border-color: rgba(15, 118, 110, 0.32);
  background: rgba(63, 81, 181, 0.06);
  color: var(--vp-c-brand-1);
}

.ct-support-link + .ct-support-link {
  margin-top: 8px;
}

.ct-support-link-main {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.ct-support-qr-card {
  display: grid;
  gap: 8px;
  justify-items: center;
  margin-top: 10px;
  color: var(--vp-c-text-2);
  font-size: 12px;
  text-align: center;
}

.ct-support-qr-zoom-button {
  display: block;
  max-width: 100%;
  padding: 0;
  border: 0;
  border-radius: 8px;
  background: transparent;
  cursor: zoom-in;
}

.ct-support-qr-zoom-button:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 3px;
}

.ct-support-qr-card img {
  display: block;
  box-sizing: border-box;
  width: min(100%, 236px);
  max-height: min(62vh, 420px);
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 8px;
  background: #fff;
  object-fit: contain;
}

.ct-support-qr-card.is-wide .ct-support-qr-zoom-button,
.ct-support-qr-card.is-wide img {
  width: 100%;
  max-width: 496px;
}

.ct-support-note {
  margin: 10px 0 0;
  color: var(--vp-c-text-2);
  font-size: 12px;
  line-height: 1.65;
}

.ct-reading-tools-fade-enter-active,
.ct-reading-tools-fade-leave-active {
  transition: opacity 0.18s cubic-bezier(0.22, 1, 0.36, 1);
}

.ct-reading-tools-fade-enter-active .ct-popover-surface {
  animation: ct-popover-surface-enter 0.18s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.ct-reading-tools-fade-leave-active .ct-popover-surface {
  animation: ct-popover-surface-leave 0.14s ease both;
}

.ct-reading-tools-fade-enter-from,
.ct-reading-tools-fade-leave-to {
  opacity: 0;
}

@keyframes ct-popover-surface-enter {
  from {
    transform: translateY(-4px) scale(0.98);
  }

  to {
    transform: translateY(0) scale(1);
  }
}

@keyframes ct-popover-surface-leave {
  from {
    transform: translateY(0) scale(1);
  }

  to {
    transform: translateY(-4px) scale(0.98);
  }
}

@media (prefers-reduced-motion: reduce) {
  .ct-reading-tools-fade-enter-active,
  .ct-reading-tools-fade-leave-active,
  .ct-reading-tools-fade-enter-active .ct-popover-surface,
  .ct-reading-tools-fade-leave-active .ct-popover-surface {
    transition: none;
  }

  .ct-reading-tools-fade-enter-active .ct-popover-surface,
  .ct-reading-tools-fade-leave-active .ct-popover-surface {
    animation: none;
  }

  .ct-popover-surface {
    transform: none;
  }
}

.ct-route-loading {
  position: fixed;
  top: 0;
  right: 0;
  left: 0;
  z-index: 1000;
  height: 3px;
  overflow: hidden;
  pointer-events: none;
  background: transparent;
}

.ct-route-loading-spinner {
  display: block;
  width: 36%;
  min-width: 120px;
  height: 100%;
  border-radius: 0 999px 999px 0;
  background: var(--vp-c-brand-1);
  box-shadow: 0 0 8px color-mix(in srgb, var(--vp-c-brand-1) 55%, transparent);
  animation: ct-route-loading-progress 0.8s ease-in-out infinite;
}

.ct-route-loading-fade-enter-active,
.ct-route-loading-fade-leave-active {
  transition: opacity 0.14s ease;
}

.ct-route-loading-fade-enter-from,
.ct-route-loading-fade-leave-to {
  opacity: 0;
}

.dark .ct-route-loading-spinner {
  background: #9aa8ff;
}

@keyframes ct-route-loading-progress {
  from {
    transform: translateX(-110%);
  }

  to {
    transform: translateX(290%);
  }
}

.ct-mobile-language-switcher {
  display: none;
  margin-top: 24px;
  padding-top: 22px;
  border-top: 1px solid var(--vp-c-divider);
}

.ct-mobile-language-title {
  font-size: 14px;
  font-weight: 500;
  color: var(--vp-c-text-1);
}

.ct-mobile-language-options {
  display: flex;
  align-items: center;
  gap: 18px;
  margin-top: 12px;
  font-size: 13px;
  line-height: 32px;
}

.ct-mobile-language-current {
  color: var(--vp-c-brand-1);
  font-weight: 500;
}

.ct-mobile-language-link {
  color: var(--vp-c-text-1);
  transition: color 0.2s;
}

.ct-mobile-language-link:hover {
  color: var(--vp-c-brand-1);
}

.ct-sidebar-hover-area {
  display: none;
  position: fixed;
  top: 0;
  left: var(--ct-sidebar-edge-right, var(--vp-sidebar-width, 280px));
  width: 24px;
  height: 100vh;
  z-index: 30;
}

.ct-sidebar-hover-area.collapsed {
  left: 0;
  width: 28px;
}

.ct-sidebar-resizer {
  position: absolute;
  left: 0;
  top: 0;
  width: 2px;
  height: 100%;
  background: var(--vp-c-divider);
  opacity: 0;
  cursor: col-resize;
  transition:
    opacity 0.2s ease,
    background-color 0.2s ease;
}

.ct-sidebar-hover-area:hover .ct-sidebar-resizer,
.ct-sidebar-hover-area.resizing .ct-sidebar-resizer {
  opacity: 1;
  background: var(--vp-c-brand-1);
}

.ct-sidebar-toggle-btn {
  position: absolute;
  top: 50%;
  left: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 38px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 0 4px 4px 0;
  background: rgba(255, 255, 255, 0.92);
  color: var(--vp-c-text-3);
  box-shadow:
    0 2px 2px 0 rgba(0, 0, 0, 0.14),
    0 3px 1px -2px rgba(0, 0, 0, 0.2),
    0 1px 5px 0 rgba(0, 0, 0, 0.12);
  cursor: pointer;
  transform: translateY(-50%);
  opacity: 0;
  transition:
    opacity 0.2s ease,
    color 0.2s ease,
    background-color 0.2s ease;
}

.ct-sidebar-hover-area:hover .ct-sidebar-toggle-btn,
.ct-sidebar-hover-area.resizing .ct-sidebar-toggle-btn {
  opacity: 1;
}

.ct-sidebar-hover-area.collapsed .ct-sidebar-toggle-btn {
  opacity: 0.92;
}

.ct-sidebar-toggle-btn:hover {
  background: var(--vp-c-bg);
  color: var(--vp-c-brand-1);
}

@media (min-width: 960px) {
  .ct-sidebar-hover-area {
    display: block;
  }
}

@media (min-width: 1440px) {
  .VPContent.has-sidebar,
  .VPNavBar.has-sidebar .content,
  .VPNavBar.has-sidebar .divider {
    transition:
      padding-left 0.26s ease,
      transform 0.26s ease;
  }
}

.ct-sidebar-resizing,
.ct-sidebar-resizing * {
  cursor: col-resize !important;
  user-select: none;
}

.ct-sidebar-resizing .VPSidebar,
.ct-sidebar-resizing .VPContent.has-sidebar,
.ct-sidebar-resizing .VPNavBar.has-sidebar .content,
.ct-sidebar-resizing .VPNavBar.has-sidebar .divider {
  transition: none !important;
}

.dark .ct-nav-tool-button,
.dark .ct-sidebar-toggle-btn {
  border-color: rgba(255, 255, 255, 0.12);
  background: rgba(30, 30, 40, 0.92);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}

.dark .ct-reading-tools-panel,
.dark .ct-support-panel {
  border-color: rgba(255, 255, 255, 0.12);
  background: rgb(30, 30, 40);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}

.dark .ct-nav-tool-button {
  border-color: transparent;
  background: transparent;
  color: rgba(245, 245, 247, 0.58);
  box-shadow: none;
}

.dark .ct-nav-tool-button:hover,
.dark .ct-nav-tool-button:focus-visible,
.dark .ct-nav-tool-button[data-state='open'] {
  border-color: rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.08);
  color: rgba(245, 245, 247, 0.84);
}

.dark .ct-reading-tools-action {
  border-color: rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.04);
}

.dark .ct-reading-tools-action.active {
  border-color: rgba(154, 168, 255, 0.34);
  background: rgba(154, 168, 255, 0.12);
}

.dark .ct-support-link,
.dark .ct-support-qr-card img {
  border-color: rgba(255, 255, 255, 0.1);
}

@media (min-width: 960px) {
  .ct-sidebar-collapsed .VPSidebar {
    display: none !important;
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .wrapper {
    padding: 0 32px;
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .container {
    max-width: calc(var(--vp-layout-max-width) - 64px);
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .title {
    position: static !important;
    width: auto !important;
    padding: 0 !important;
    background: transparent !important;
  }

  .ct-sidebar-collapsed .VPNavBarTitle.has-sidebar .title {
    border-bottom-color: transparent !important;
  }

  .ct-sidebar-collapsed .VPContent.has-sidebar,
  .ct-sidebar-collapsed .VPNavBar.has-sidebar .content,
  .ct-sidebar-collapsed .VPNavBar.has-sidebar .divider {
    padding-left: 0 !important;
  }
}

@media (min-width: 1440px) {
  .ct-sidebar-collapsed .VPContent.has-sidebar {
    padding-left: calc(
      (100% - var(--vp-layout-max-width, 1440px)) / 2
    ) !important;
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .wrapper {
    padding: 0;
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .container {
    max-width: var(--vp-layout-max-width);
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .content,
  .ct-sidebar-collapsed .VPNavBar.has-sidebar .divider {
    padding-left: calc(
      (100% - var(--vp-layout-max-width, 1440px)) / 2
    ) !important;
  }
}

.VPHomeHero .tagline {
  display: none !important;
}

.ct-home-typed-tagline {
  margin-top: 10px;
  min-height: 30px;
  font-size: 18px;
  line-height: 1.6;
  font-weight: 500;
  color: var(--vp-c-text-2);
  text-align: center;
  white-space: pre-wrap;
}

.VPHomeHero .container,
.VPHomeHero .main,
.VPHomeHero .name,
.VPHomeHero .text,
.VPHomeHero .actions {
  text-align: center;
}

.VPHomeHero .actions {
  justify-content: center;
}

@media (min-width: 960px) {
  .ct-home-typed-tagline {
    font-size: 22px;
    line-height: 1.7;
  }
}

@media (max-width: 768px) {
  .ct-nav-tools {
    margin-left: 4px;
  }

  .ct-support-panel,
  .ct-reading-tools-panel {
    right: -6px;
    width: min(280px, calc(100vw - 24px));
  }

  .ct-mobile-language-switcher {
    display: block;
  }
}

/* Use the compact screen menu only on tablet-sized viewports. Desktop and
   laptop widths keep the course shortcuts visible in the navigation bar. */
@media (min-width: 768px) and (max-width: 1099px) {
  .VPNavBar .VPNavBarMenu,
  .VPNavBar .VPNavBarExtra {
    display: none !important;
  }

  .VPNavBar .VPNavBarHamburger {
    display: flex !important;
  }

  .VPNavScreen {
    display: block !important;
  }
}

@media (max-width: 480px) {
  .VPNavBar .container > .title {
    width: var(--vp-nav-logo-height);
  }

  .VPNavBarTitle .title {
    padding-right: 0;
  }

  .VPNavBarTitle .logo {
    margin-right: 0;
  }

  .VPNavBarTitle .ct-nav-title-text {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
    border: 0;
  }
}

.medium-zoom-overlay {
  z-index: 999;
}

.medium-zoom--opened .medium-zoom-overlay {
  background: rgba(15, 23, 42, 0.62) !important;
}

.medium-zoom-image--opened {
  z-index: 1000;
}

.ct-mermaid-viewer-open {
  overflow: hidden;
}

.ct-mermaid-viewer {
  position: fixed;
  inset: 0;
  z-index: 1001;
  display: grid;
  grid-template-rows: auto 1fr;
  background: rgba(15, 23, 42, 0.62);
}

.ct-mermaid-viewer-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
  padding: 10px 12px;
}

.ct-mermaid-viewer-help {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 10px;
  color: rgba(255, 255, 255, 0.9);
  font-size: 12px;
  line-height: 1.4;
}

.ct-mermaid-viewer-help strong {
  flex: 0 0 auto;
  min-width: 44px;
  color: #fff;
  font-weight: 700;
}

.ct-mermaid-viewer-help span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ct-mermaid-viewer-actions {
  display: flex;
  flex: 0 0 auto;
  gap: 8px;
}

.ct-mermaid-viewer-actions button {
  height: 34px;
  min-width: 44px;
  padding: 0 12px;
  border: 1px solid rgba(255, 255, 255, 0.22);
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.92);
  color: #111827;
  font-size: 13px;
  cursor: pointer;
}

.ct-mermaid-viewer-actions button:hover {
  background: #fff;
}

.ct-mermaid-viewer-scroll {
  overflow: auto;
  padding: 24px;
  cursor: grab;
  overscroll-behavior: contain;
  user-select: none;
}

.ct-mermaid-viewer-scroll.is-dragging {
  cursor: grabbing;
}

.ct-mermaid-viewer-stage {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 100%;
  min-height: 100%;
  margin: 0 auto;
}

.ct-mermaid-viewer-image {
  display: block;
  width: auto;
  max-width: none;
  max-height: none;
  height: auto;
  background: #fff;
  cursor: inherit;
  user-select: none;
  -webkit-user-drag: none;
  touch-action: none;
}

.main img {
  cursor: zoom-in;
  transition: transform 0.2s ease;
}

.main img:not(.ct-mermaid-viewer-image):hover {
  transform: scale(1.01);
}

@media (max-width: 640px) {
  .ct-mermaid-viewer-toolbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .ct-mermaid-viewer-help span {
    white-space: normal;
  }

  .ct-mermaid-viewer-actions {
    width: 100%;
  }

  .ct-mermaid-viewer-actions button {
    flex: 1 1 0;
  }
}

/* 侧边栏常驻工具栏 */
.ct-sidebar-groups-toolbar {
  position: relative;
  z-index: 2;
  order: -1;
  flex: 0 0 auto;
  margin: 0 16px 0 20px;
  padding: 4px 0 5px;
  border-bottom: 0;
  background: var(--vp-c-bg);
}

@media (min-width: 960px) {
  .ct-sidebar-groups-toolbar {
    margin-right: 32px;
    margin-left: 32px;
  }
}

/* VitePress shows its 48px local-outline bar between 960px and 1279px.
   Match that row exactly so its divider cannot cut through sidebar content. */
@media (min-width: 960px) and (max-width: 1279px) {
  .ct-sidebar-groups-toolbar {
    min-height: 48px;
    padding-top: 10px;
    padding-bottom: 10px;
  }
}

.ct-sidebar-groups-control {
  display: flex;
  align-items: center;
  justify-content: center;
}

.ct-sidebar-toolbar-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 2px 4px;
}

.ct-sidebar-groups-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  gap: 5px;
  min-width: 94px;
  height: 28px;
  padding: 0 4px;
  color: var(--vp-c-text-2);
  background: transparent;
  border: 0;
  border-radius: 7px;
  font-family: inherit;
  font-size: 12px;
  font-weight: 540;
  letter-spacing: 0.005em;
  white-space: nowrap;
  cursor: pointer;
  transition:
    color 0.2s,
    background-color 0.2s,
    transform 0.2s;
}

.ct-sidebar-groups-button svg {
  color: var(--vp-c-brand-1);
  opacity: 0.58;
  transition:
    opacity 0.2s,
    transform 0.2s;
}

.ct-sidebar-groups-button:hover {
  color: var(--vp-c-text-1);
  background-color: color-mix(in srgb, var(--vp-c-bg-soft) 76%, transparent);
}

.ct-sidebar-groups-button:hover svg {
  opacity: 0.92;
  transform: translateY(1px);
}

.ct-sidebar-groups-button.is-collapse-action:hover svg {
  transform: translateY(-1px);
}

.ct-sidebar-groups-button:active {
  background-color: var(--vp-c-brand-soft);
  transform: scale(0.98);
}

.ct-sidebar-groups-button:focus-visible {
  outline: 2px solid color-mix(in srgb, var(--vp-c-brand-1) 68%, transparent);
  outline-offset: 1px;
}

@media (max-width: 959px) {
  .ct-sidebar-groups-toolbar {
    padding-top: 3px;
    padding-bottom: 4px;
  }

  .ct-sidebar-groups-button {
    min-width: 100px;
    height: 30px;
    padding-right: 6px;
    padding-left: 6px;
    font-size: 12.5px;
  }

  .ct-sidebar-toolbar-actions .ct-sidebar-footer-btn,
  .ct-sidebar-toolbar-actions .ct-sidebar-footer-link {
    width: 30px;
    height: 30px;
  }
}
</style>
