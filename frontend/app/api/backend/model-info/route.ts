import { proxyToFastAPI } from '@/lib/fraudshield-proxy'

export const dynamic = 'force-dynamic'

export function GET() {
  return proxyToFastAPI({ path: '/model-info', method: 'GET', timeoutMs: 8_000 })
}
