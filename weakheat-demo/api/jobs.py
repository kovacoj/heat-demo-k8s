"""Kubernetes Job management (spec section 26).

Everything is hard-coded: image, resources, command.  Only the four
validated physical parameters are interpolated into the fixed command.
At most MAX_CONCURRENT Firedrake Jobs are allowed; extra requests are
rejected with 429 by the API layer.
"""
import os
import re
import uuid

from kubernetes import client as k8s
from kubernetes import config as k8s_config

FDMI_IMAGE = os.environ.get("WEAKHEAT_FDMI_IMAGE",
                            "cerit.io/kovacoj1/weakheat-firedrake:dev")
NAMESPACE = os.environ.get("POD_NAMESPACE", "kovacovsky-ns")
CALLBACK_HOST = os.environ.get("WEAKHEAT_CALLBACK_HOST", "weakheat-api-svc")
MAX_CONCURRENT = 2
JOB_TTL_S = 600
JOB_DEADLINE_S = 900

JOB_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def make_job(job_id: str, p: dict, callback_token: str) -> k8s.V1Job:
    name = f"weakheat-fd-{job_id}"
    assert JOB_NAME_RE.match(name)
    command = [
        "python3", "/app/worker.py",
        "--job-id", job_id,
        "--x0", repr(p["x0"]),
        "--y0", repr(p["y0"]),
        "--sigma", repr(p["sigma"]),
        "--alpha", repr(p["alpha"]),
        "--callback-host", CALLBACK_HOST,
    ]
    container = k8s.V1Container(
        name="firedrake",
        image=FDMI_IMAGE,
        command=command,
        env=[k8s.V1EnvVar(name="CALLBACK_TOKEN", value=callback_token),
             k8s.V1EnvVar(name="HOME", value="/tmp")],
        resources=k8s.V1ResourceRequirements(
            requests={"cpu": "1", "memory": "1Gi"},
            limits={"cpu": "2", "memory": "2Gi"},
        ),
        security_context=k8s.V1SecurityContext(
            run_as_non_root=True, run_as_user=1000,
            allow_privilege_escalation=False,
            capabilities=k8s.V1Capabilities(drop=["ALL"]),
            seccomp_profile=k8s.V1SeccompProfile(type="RuntimeDefault"),
        ),
    )
    template = k8s.V1PodTemplateSpec(
        metadata=k8s.V1ObjectMeta(labels={"app": "weakheat-firedrake", "sim": name}),
        spec=k8s.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            security_context=k8s.V1PodSecurityContext(
                run_as_non_root=True,
                seccomp_profile=k8s.V1SeccompProfile(type="RuntimeDefault"),
            ),
        ),
    )
    spec = k8s.V1JobSpec(
        template=template,
        backoff_limit=0,
        ttl_seconds_after_finished=JOB_TTL_S,
        active_deadline_seconds=JOB_DEADLINE_S,
    )
    return k8s.V1Job(
        metadata=k8s.V1ObjectMeta(name=name, labels={"app": "weakheat-firedrake"}),
        spec=spec,
    )


class JobManager:
    def __init__(self):
        self.available = False
        try:
            k8s_config.load_incluster_config()
            self.available = True
        except Exception:  # local testing outside the cluster
            try:
                k8s_config.load_kube_config()
                self.available = True
            except Exception:  # noqa: BLE001
                pass
        self.batch = k8s.BatchV1Api()
        self.core = k8s.CoreV1Api()

    def new_job_id(self) -> str:
        return uuid.uuid4().hex[:6]

    def active_jobs(self) -> int:
        """Jobs created by this API that are not finished."""
        if not self.available:
            return 0
        jobs = self.batch.list_namespaced_job(
            namespace=NAMESPACE, label_selector="app=weakheat-firedrake")
        n = 0
        for j in jobs.items:
            if j.status.completion_time is None and not (j.status.failed or 0):
                n += 1
        return n

    def create_job(self, job_id: str, p: dict, callback_token: str):
        job = make_job(job_id, p, callback_token)
        self.batch.create_namespaced_job(namespace=NAMESPACE, body=job)
        return job.metadata.name

    def job_phase(self, job_id: str) -> str | None:
        """'running' if any pod of the job is Running, else None."""
        if not self.available:
            return None
        try:
            pods = self.core.list_namespaced_pod(
                namespace=NAMESPACE, label_selector=f"sim=weakheat-fd-{job_id}")
        except Exception:  # noqa: BLE001
            return None
        for pod in pods.items:
            if pod.status.phase == "Running":
                return "running"
        return None
