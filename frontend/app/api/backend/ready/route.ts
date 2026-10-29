import { proxyToFastAPI } from '@/lib/fraudshield-proxy'

export const dynamic = 'force-dynamic'

export function GET() {
  return proxyToFastAPI({ path: '/ready', method: 'GET', timeoutMs: 8_000 })
}
