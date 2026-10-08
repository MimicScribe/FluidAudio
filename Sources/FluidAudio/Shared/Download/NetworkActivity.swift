import Foundation

/// Every HTTP request the download stack makes, reported to the host app (for
/// a network log). One report per hop: a redirect to a CDN host is its own
/// entry, so the host sees every server contacted. Responses served from the
/// local URL cache are not reported; nothing left the machine for them.
public enum NetworkActivity {
    public struct Request: Sendable {
        public let url: URL?
        public let method: String
        public let statusCode: Int?
        public let startedAt: Date?
        public let responseBytes: Int?
        public let errorDescription: String?
    }

    /// Set once at startup, before any download starts.
    nonisolated(unsafe) public static var observer: (@Sendable (Request) -> Void)?

    /// A session whose requests are reported.
    static func session(configuration: URLSessionConfiguration) -> URLSession {
        URLSession(configuration: configuration, delegate: MetricsDelegate.shared, delegateQueue: nil)
    }

    static func report(task: URLSessionTask, metrics: URLSessionTaskMetrics) {
        guard let observer else { return }
        let hops = metrics.transactionMetrics.filter { $0.resourceFetchType == .networkLoad }
        for (index, hop) in hops.enumerated() {
            let isLast = index == hops.count - 1
            observer(
                Request(
                    url: hop.request.url,
                    method: hop.request.httpMethod ?? "GET",
                    statusCode: (hop.response as? HTTPURLResponse)?.statusCode,
                    startedAt: hop.fetchStartDate,
                    responseBytes: Int(hop.countOfResponseBodyBytesReceived),
                    errorDescription: isLast ? task.error?.localizedDescription : nil))
        }
        // Failed before any hop was recorded (DNS, refused connection).
        if hops.isEmpty, let error = task.error {
            observer(
                Request(
                    url: task.originalRequest?.url, method: task.originalRequest?.httpMethod ?? "GET",
                    statusCode: nil, startedAt: metrics.taskInterval.start, responseBytes: nil,
                    errorDescription: error.localizedDescription))
        }
    }

    private final class MetricsDelegate: NSObject, URLSessionTaskDelegate, Sendable {
        static let shared = MetricsDelegate()

        func urlSession(
            _ session: URLSession, task: URLSessionTask, didFinishCollecting metrics: URLSessionTaskMetrics
        ) {
            NetworkActivity.report(task: task, metrics: metrics)
        }
    }
}
