import DefaultTheme from 'vitepress/theme'
import 'katex/dist/katex.min.css'
import './style.css'
import DpoCodeFocus from '../../chapter17_dpo/components/DpoCodeFocus.vue'
import Layout from './Layout.vue'
import CourseNavMenu from './components/CourseNavMenu.vue'
import ChapterNotebookLink from './components/ChapterNotebookLink.vue'
import NavCard from './components/NavCard.vue'
import NavGrid from './components/NavGrid.vue'
import OnlineTraining from './components/OnlineTraining.vue'
import PpoCodeFocus from '../../chapter10_ppo/components/PpoCodeFocus.vue'
import GrpoCodeFocus from '../../chapter18_grpo/components/GrpoCodeFocus.vue'
import StepBar from './components/StepBar.vue'

function loadFonts() {
  // Apple 风格：使用系统原生字体，不加载外部字体
  // SF Pro / PingFang SC / Hiragino Sans GB 等系统字体已足够精美
  if (typeof document === 'undefined') return
}

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp(ctx) {
    DefaultTheme.enhanceApp?.(ctx)
    ctx.app.component('DpoCodeFocus', DpoCodeFocus)
    ctx.app.component('GrpoCodeFocus', GrpoCodeFocus)
    ctx.app.component('NavCard', NavCard)
    ctx.app.component('NavGrid', NavGrid)
    ctx.app.component('OnlineTraining', OnlineTraining)
    ctx.app.component('PpoCodeFocus', PpoCodeFocus)
    ctx.app.component('StepBar', StepBar)
    ctx.app.component('CourseNavMenu', CourseNavMenu)
    ctx.app.component('ChapterNotebookLink', ChapterNotebookLink)
    loadFonts()
  }
}
