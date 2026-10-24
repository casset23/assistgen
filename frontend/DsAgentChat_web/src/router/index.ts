import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Login from '../views/Login.vue'
import EcommerceService from '../views/EcommerceService.vue'
import KnowledgeBaseView from '../views/KnowledgeBaseView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/preview',
      name: 'preview',
      component: KnowledgeBaseView,
      props: { preview: true }
    },
    {
      path: '/',
      redirect: '/chat'
    },
    {
      path: '/chat',
      name: 'home',
      component: Home,
      meta: { requiresAuth: true }
    },
    {
      path: '/knowledge',
      name: 'knowledge',
      component: KnowledgeBaseView,
      meta: { requiresAuth: true }
    },
    {
      path: '/login',
      name: 'login',
      component: Login
    },
    {
      path: '/register',
      name: 'register',
      component: Login
    },
    {
      path: '/ecommerce',
      name: 'ecommerce',
      component: EcommerceService,
      meta: { requiresAuth: true }
    }
  ]
})

router.beforeEach((to, _from, next) => {
  if (to.name === 'preview') {
    next()
    return
  }
  const token = localStorage.getItem('token')

  if ((to.path === '/login' || to.path === '/register') && token) {
    next('/chat')
    return
  }

  if (!token && to.path !== '/login' && to.path !== '/register') {
    next('/login')
    return
  }

  next()
})

export default router
