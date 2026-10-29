import { proxyToFastAPI } from '@/lib/fraudshield-proxy'

export const dynamic = 'force-dynamic'

export function GET() {
  return proxyToFastAPI({ path: '/roi', method: 'GET', timeoutMs: 8_000 })
}

export function POST(request: Request) {
  return proxyToFastAPI({ path: '/roi', method: 'POST', request, protectedRoute: true, timeoutMs: 15_000 })
}
