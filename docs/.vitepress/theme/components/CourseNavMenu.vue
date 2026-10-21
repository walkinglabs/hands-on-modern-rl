<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useData, useRoute, withBase } from 'vitepress'
import { ChevronDown } from 'lucide-vue-next'

const props = defineProps({
  text: { type: String, required: true },
  items: { type: Array, required: true },
  activeMatch: { type: String, default: '' },
  screenMenu: { type: Boolean, default: false }
})
const { site } = useData()
const route = useRoute()
const menu = ref(null)
const currentPath = computed(
  () => '/' + route.path.slice(site.value.base.length)
)
const active = computed(
  () =>
    props.activeMatch && new RegExp(props.activeMatch).test(currentPath.value)
)

function close(restoreFocus = false) {
  if (!menu.value?.open) return
  menu.value.open = false
  if (restoreFocus) menu.value.querySelector('summary').focus()
}
function outside(event) {
  if (!menu.value?.contains(event.target)) close()
}
function closePeers() {
  if (!menu.value?.open) return
  document.querySelectorAll('.ct-course-menu[open]').forEach((other) => {
    if (other !== menu.value) other.open = false
  })
}
onMounted(() => document.addEventListener('click', outside))
onBeforeUnmount(() => document.removeEventListener('click', outside))
</script>

<template>
  <details
    ref="menu"
    class="ct-course-menu"
    :class="{ 'is-active': active, 'is-screen': screenMenu }"
    @toggle="closePeers"
    @keydown.esc.stop.prevent="close(true)"
  >
    <summary>
      {{ text }}
      <ChevronDown :size="12" aria-hidden="true" />
    </summary>
    <ul class="ct-course-links">
      <li v-for="item in items" :key="item.link">
        <a
          :href="withBase(item.link)"
          :aria-current="currentPath === item.link ? 'page' : undefined"
          @click="close()"
        >
          {{ item.text }}
        </a>
      </li>
    </ul>
  </details>
</template>

<style scoped>
.ct-course-menu {
  position: relative;
  display: flex;
  align-content: center;
  margin: auto 0;
  color: var(--vp-c-text-2);
}
.ct-course-menu summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 7px;
  min-height: 36px;
  padding: 7px 12px;
  border-radius: 18px;
  font-size: 13.5px;
  font-weight: 500;
  line-height: 22px;
  list-style: none;
  cursor: pointer;
  user-select: none;
}
.ct-course-menu summary::-webkit-details-marker {
  display: none;
}
.ct-course-menu summary:hover,
.ct-course-menu[open] summary {
  background: var(--vp-c-default-soft);
  color: var(--vp-c-text-1);
}
.ct-course-menu.is-active summary {
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}
.ct-course-menu summary svg {
  opacity: 0.7;
}
.ct-course-menu[open] summary svg {
  transform: rotate(180deg);
}
.ct-course-links {
  position: absolute;
  right: 0;
  top: calc(100% + 8px);
  min-width: 180px;
  padding: 7px;
  margin: 0;
  list-style: none;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-elv);
  box-shadow: var(--vp-shadow-3);
}
.ct-course-links a {
  display: block;
  padding: 10px 12px;
  border-radius: 6px;
  font-size: 13px;
  line-height: 20px;
  white-space: nowrap;
  color: var(--vp-c-text-1);
}
.ct-course-links a:hover,
.ct-course-links a[aria-current='page'] {
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}
.is-screen {
  display: block;
  width: 100%;
  border-bottom: 1px solid var(--vp-c-divider);
}
.is-screen summary {
  min-height: 52px;
  padding: 12px 0;
  border-radius: 0;
  font-size: 14px;
}
.is-screen summary:hover,
.is-screen[open] summary {
  background: transparent;
}
.is-screen .ct-course-links {
  position: static;
  min-width: 0;
  padding: 0 0 12px 12px;
  border: 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
}
.is-screen .ct-course-links a {
  white-space: normal;
  min-height: 44px;
  padding: 12px;
}
</style>
