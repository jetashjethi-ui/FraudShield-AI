import { proxyToFastAPI } from '@/lib/fraudshield-proxy'

export const dynamic = 'force-dynamic'
export const maxDuration = 120

export function POST(request: Request) {
  return proxyToFastAPI({
    path: '/batch',
    method: 'POST',
    request,
    protectedRoute: true,
    timeoutMs: 110_000,
    maxBodyBytes: 5 * 1024 * 1024,
  })
}
