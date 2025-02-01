# Standard
from datetime import datetime

# First Party
from instructlab.defaults import ILAB_PROCESS_STATUS
from instructlab.process import process


class TestProcessRegistry:
    @staticmethod
    def _get_registry(home):
        return process.ProcessRegistry(filepath=home / "process_registry.json").load()

    def test_add_update_remove(self, tmp_path_home):
        reg = self._get_registry(tmp_path_home)
        assert not reg.processes

        # Add a process
        uuid_ = "fake-uuid"
        pid = 100
        children = [101, 102]
        type_ = "TestType"
        log_file = tmp_path_home / "test.log"
        now = datetime.now()
        status = ILAB_PROCESS_STATUS.RUNNING

        reg.add(
            uuid_,
            process.Process(
                pid=pid,
                log_path=log_file,
                ptype=type_,
                children=children[:],
                start_time=now,
                status=status,
            ),
        )

        # Confirm values are as expected
        assert len(reg.processes) == 1
        p = reg.processes[uuid_]
        assert p.pids == [100, 101, 102]
        assert p.ptype == type_
        assert p.log_path == log_file
        runtime = p.runtime
        assert runtime <= datetime.now() - now
        assert p.status == status
        assert not p.completed
        assert not p.started

        # Create log file; this tells the module that the process started
        log_file.touch()
        assert not p.completed
        assert p.started

        # Now complete the process
        p.complete(ILAB_PROCESS_STATUS.DONE)

        # Confirm update happened
        p = reg.processes[uuid_]
        assert p.started
        assert p.completed

        # Remove the process and confirm it's removed
        reg.remove(uuid_)
        assert not reg.processes

    def test_load_persist(self, tmp_path_home):
        reg = self._get_registry(tmp_path_home)
        assert not reg.processes

        # Add a process
        uuid_ = "fake-uuid"
        pid = 100
        children = [101, 102]
        type_ = "TestType"
        log_file = tmp_path_home / "test.log"
        now = datetime.now()
        status = ILAB_PROCESS_STATUS.RUNNING

        reg.add(
            uuid_,
            process.Process(
                pid=pid,
                log_path=log_file,
                ptype=type_,
                children=children[:],
                start_time=now,
                status=status,
            ),
        )

        # Persist the registry
        reg.persist()

        # Load a new registry and confirm it's the same
        new_reg = self._get_registry(tmp_path_home)
        assert len(new_reg.processes) == 1
        p = new_reg.processes[uuid_]
        assert p.pids == [100, 101, 102]
        assert p.ptype == type_
        assert p.log_path == log_file
        assert p._start_time == now
        assert p.status == status
