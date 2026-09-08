IMAGE := cerit.io/kovacoj1/heat-firedrake:dev

NAMESPACE := kovacovsky-ns

DEPLOYMENT := heat-demo


.PHONY: build push deploy status logs restart


build:
	docker build -t $(IMAGE) .


push:
	docker push $(IMAGE)


deploy: build push
	kubectl apply -f deployment.yaml -n $(NAMESPACE)
	kubectl apply -f service.yaml -n $(NAMESPACE)
	kubectl apply -f ingress.yaml -n $(NAMESPACE)
	kubectl rollout restart deployment/$(DEPLOYMENT) -n $(NAMESPACE)
	kubectl rollout status deployment/$(DEPLOYMENT) -n $(NAMESPACE)


restart:
	kubectl rollout restart deployment/$(DEPLOYMENT) -n $(NAMESPACE)


status:
	kubectl get pods -n $(NAMESPACE)
	kubectl get service -n $(NAMESPACE)
	kubectl get ingress -n $(NAMESPACE)


logs:
	kubectl logs -f deployment/$(DEPLOYMENT) -n $(NAMESPACE)
