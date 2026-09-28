# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg538::mock.Mock+mock.patch+wooey.tasks.cleanup_dead_jobs
# name: mock_wooey_primitive
# summary: Uses mock.Mock, mock.patch, wooey.tasks.cleanup_dead_jobs across 2 repos
# anchor_symbols: ['mock.Mock', 'mock.patch', 'wooey.tasks.cleanup_dead_jobs']
# observed in 2 repos: ['HDI-Project__ATM', 'wooey__Wooey']...

# --- from HDI-Project__ATM::tests/test_worker.py::test_tune_hyperparameters ---
def test_tune_hyperparameters(worker, hyperpartition):
    """
    This won't test that BTB is working correctly, just that the ATM-BTB
    connection is working.
    """
    mock_tuner = Mock()
    worker.Tuner = Mock(return_value=mock_tuner)

    with patch('atm.worker.update_params') as update_params_mock:
        worker.tune_hyperparameters(hyperpartition)

        update_params_mock.assert_called_once_with(
            params=mock_tuner.propose.return_value,
            categoricals=hyperpartition.categoricals,
            constants=hyperpartition.constants
        )

    mock_tuner.propose.assert_called()

# --- from wooey__Wooey::wooey/tests/test_tasks.py::TestCleanupDeadJobs.test_handles_unresponsive_workers ---
def test_handles_unresponsive_workers(self):
        # Ensure that if we cannot connect to celery, we do nothing.
        with mock.patch("wooey.tasks.celery_app.control.inspect") as inspect_mock:
            running_job = factories.generate_job(self.translate_script)
            running_job.status = WooeyJob.RUNNING
            running_job.save()

            inspect_mock.return_value = mock.Mock(
                active=mock.Mock(
                    return_value=None,
                )
            )
            cleanup_dead_jobs()
            self.assertEqual(
                WooeyJob.objects.get(pk=running_job.id).status, WooeyJob.RUNNING
            )

# --- from wooey__Wooey::wooey/tests/test_tasks.py::TestCleanupDeadJobs.test_cleans_up_dead_jobs ---
def test_cleans_up_dead_jobs(self):
        # Make a job that is running but not active, and a job that is running and active.
        dead_job = factories.generate_job(self.translate_script)
        dead_job.status = WooeyJob.RUNNING
        dead_job.save()
        active_job = factories.generate_job(self.translate_script)
        active_job.status = WooeyJob.RUNNING
        active_job.celery_id = "celery-id"
        active_job.save()
        with mock.patch("wooey.tasks.celery_app.control.inspect") as inspect_mock:
            inspect_mock.return_value = mock.Mock(
                active=mock.Mock(
                    return_value={
                        "worker-id": [
                            {
                                "id": active_job.celery_id,
                            }
                        ]
                    },
                )
            )
            cleanup_dead_jobs()

            # Assert the dead job is updated
            self.assertEqual(
                WooeyJob.objects.get(pk=dead_job.id).status, WooeyJob.FAILED
            )
            self.assertEqual(
                WooeyJob.objects.get(pk=active_job.id).status, WooeyJob.RUNNING
            )
