<script setup>
import { withBase } from 'vitepress'
import { resolveLabSection } from './lab-sections.js'
import { FlaskConical, ArrowRight } from 'lucide-vue-next'
defineProps({
  locale: { type: String, default: 'zh' },
  chapter: { type: String, default: 'intro' }
})
</script>

<template>
  <a
    class="chapter-notebook-entry"
    :href="
      withBase(
        (locale === 'en' ? '/en/labs/' : '/labs/') +
          '?section=' +
          resolveLabSection(chapter)
      )
    "
  >
    <FlaskConical :size="16" aria-hidden="true" />
    <span>
      {{
        locale === 'en'
          ? 'Browse related experiments in the lab'
          : '在实验室浏览相关实验'
      }}
    </span>
    <ArrowRight :size="16" aria-hidden="true" />
  </a>
</template>

<style scoped>
.chapter-notebook-entry {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 24px 0;
  padding: 16px 20px;
  background: var(--vp-c-brand-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  color: var(--vp-c-brand-1);
  font-size: 14px;
  font-weight: 500;
  text-decoration: none;
  min-height: 52px;
}
.chapter-notebook-entry > svg {
  flex-shrink: 0;
}
.chapter-notebook-entry > svg:last-child {
  margin-left: auto;
}
</style>
