def start_heartbeat(self, receipt_handle: str) -> Heartbeat:
        visibility_timeout = self.visibility_timeout
        if visibility_timeout is None:
            raise RuntimeError("visibility timeout must be resolved before starting a heartbeat")
        heartbeat = Heartbeat(
            self._client,
            receipt_handle,
            interval_seconds=max(1, visibility_timeout) / 3,
            visibility_timeout=visibility_timeout,
            max_runtime_seconds=self.max_runtime_seconds,
        )
        # A zero visibility timeout keeps messages visible, so there is no visibility window to extend.
        if visibility_timeout > 0:
            heartbeat.start()
        return heartbeat