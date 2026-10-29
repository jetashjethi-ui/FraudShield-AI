import { proxyToFastAPI } from '@/lib/fraudshield-proxy'

export const dynamic = 'force-dynamic'

export function POST(request: Request) {
  return proxyToFastAPI({ path: '/score', method: 'POST', request, protectedRoute: true, timeoutMs: 20_000 })
}
