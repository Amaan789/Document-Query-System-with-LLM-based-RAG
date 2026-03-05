
NETWORK = ""


clean: ##=> Deletes current build environment and latest build
	$(info [*] Who needs all that anyway? Destroying environment....)
	rm -rf ./.aws-sam/

deploy.guided: ##=> Guided deploy that is typically run for the first time only
	sam build --use-container -t template.yml
	sam deploy --guided --capabilities CAPABILITY_IAM --profile default --config-file samconfig.toml

up:
	pip3 install -r requirements.txt
	uvicorn main:app --reload
