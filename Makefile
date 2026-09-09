REGISTRY := cerit.io/kovacoj1/heat-firedrake

OF_REGISTRY := cerit.io/kovacoj1/heat-openfoam

NAMESPACE := kovacovsky-ns

DEPLOYMENT := heat-demo

CONTAINER := heat-demo


# Unique tag per deploy: the Harbor registry sometimes serves
# a stale manifest for a re-used mutable tag right after a push,
# so "kubectl rollout restart" can pull the previous image.
TAG := $(shell date +%Y%m%d-%H%M%S)


.PHONY: build push of-build of-push deploy status logs restart


build:
	docker build -t $(REGISTRY):$(TAG) .


push:
	docker push $(REGISTRY):$(TAG)


of-build:
	docker build -t $(OF_REGISTRY):$(TAG) openfoam


of-push:
	docker push $(OF_REGISTRY):$(TAG)


deploy: build push of-build of-push
	kubectl apply -f service.yaml -n $(NAMESPACE)
	kubectl apply -f ingress.yaml -n $(NAMESPACE)
	kubectl apply -f rbac.yaml -n $(NAMESPACE)
	kubectl apply -f deployment.yaml -n $(NAMESPACE)
	kubectl set image deployment/$(DEPLOYMENT) \
		$(CONTAINER)=$(REGISTRY):$(TAG) \
		-n $(NAMESPACE)
	kubectl set env deployment/$(DEPLOYMENT) \
		SIM_IMAGE=$(REGISTRY):$(TAG) \
		OF_IMAGE=$(OF_REGISTRY):$(TAG) \
		-n $(NAMESPACE)
	kubectl rollout status deployment/$(DEPLOYMENT) -n $(NAMESPACE)


restart:
	kubectl rollout restart deployment/$(DEPLOYMENT) -n $(NAMESPACE)


status:
	kubectl get pods -n $(NAMESPACE)
	kubectl get service -n $(NAMESPACE)
	kubectl get ingress -n $(NAMESPACE)


logs:
	kubectl logs -f deployment/$(DEPLOYMENT) -n $(NAMESPACE)
