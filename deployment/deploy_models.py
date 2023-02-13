import argparse
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice


def get_model_latest_version(ws: Workspace, model_name: str) -> str:
    all_versions = [
        (model.id, model.version)
        for model in Model.list(ws)
        if model.id.startswith(model_name)
    ]
    all_versions.sort(key=lambda x: x[1], reverse=True)
    return all_versions[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws_name", type=str, help="AML workspace name")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription id")
    parser.add_argument("--rg_name", type=str, help="Resource group name")
    parser.add_argument("--tenant_id", type=str, help="Tenant id")
    parser.add_argument("--service_principal_id", type=str, help="Application id")
    parser.add_argument("--service_name", type=str, help="Endpoint service name")
    parser.add_argument("--model_1", type=str, help="Name of the first model")
    parser.add_argument("--model_2", type=str, help="Name of the second model")
    parser.add_argument("--model_3", type=str, help="Name of the third model")
    parser.add_argument(
        "--azure_secret", type=str, help="Service Principal Scecret key"
    )
    args = parser.parse_args()

    svc_pr = ServicePrincipalAuthentication(
        tenant_id=args.tenant_id,
        service_principal_id=args.service_principal_id,
        service_principal_password=args.azure_secret,
    )

    ws = Workspace(
        subscription_id=args.subscription_id,
        resource_group=args.rg_name,
        workspace_name=args.ws_name,
        auth=svc_pr,
    )

    print("Found workspace {} at location {}".format(ws.name, ws.location))

    model_1 = Model(ws, id=get_model_latest_version(ws, args.model_1))
    model_2 = Model(ws, id=get_model_latest_version(ws, args.model_2))
    model_3 = Model(ws, id=get_model_latest_version(ws, args.model_3))

    env = Environment.from_dockerfile(
        "keystrokes_env", "Dockerfile", conda_specification=None, pip_requirements=None
    )
    env.inferencing_stack_version = "latest"
    inference_config = InferenceConfig(entry_script="score.py", environment=env)

    aci_service_name = args.service_name

    deployment_config = AciWebservice.deploy_configuration(cpu_cores=2, memory_gb=1)

    service = Model.deploy(
        ws,
        aci_service_name,
        [model_1, model_2, model_3],
        inference_config,
        deployment_config,
        overwrite=True,
    )
    service.wait_for_deployment(True)

    print(service.state)


if __name__ == "__main__":
    main()
