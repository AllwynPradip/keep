"""
Grafana Provider is a class that allows to ingest/digest data from Grafana.
"""

import dataclasses
import datetime
import hashlib
import json
import logging
import time
from functools import wraps
from typing import Dict, Optional, Callable, Any

import pydantic
import requests
from packaging.version import Version

from keep.api.models.alert import AlertDto, AlertSeverity, AlertStatus
from keep.api.models.db.topology import TopologyServiceInDto
from keep.contextmanager.contextmanager import ContextManager
from keep.providers.base.base_provider import (
    BaseProvider,
    BaseTopologyProvider,
    ProviderHealthMixin,
)
from keep.providers.base.provider_exceptions import GetAlertException
from keep.providers.grafana_provider.grafana_alert_format_description import (
    GrafanaAlertFormatDescription,
)
from keep.providers.models.provider_config import ProviderConfig, ProviderScope
from keep.providers.providers_factory import ProvidersFactory

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (requests.exceptions.RequestException, requests.exceptions.Timeout, requests.exceptions.ConnectionError)
):
    """
    Decorator that implements exponential backoff retry logic for functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor by which delay increases each retry
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries",
                            extra={"error": str(e), "attempt": attempt + 1}
                        )
                        raise e
                    
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}, retrying in {delay}s",
                        extra={"error": str(e), "delay": delay}
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                except Exception as e:
                    # Don't retry on non-network exceptions
                    logger.error(
                        f"Function {func.__name__} failed with non-retryable exception",
                        extra={"error": str(e), "attempt": attempt + 1}
                    )
                    raise e
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class GrafanaConnectionManager:
    """
    Manages HTTP connections to Grafana with session reuse and validation.
    """
    
    def __init__(self, host: str, token: str, timeout: int = 30):
        self.host = host
        self.token = token
        self.timeout = timeout
        self._session = None
        self._session_created_at = None
        self._session_max_age = 3600  # 1 hour session lifetime
    
    def get_session(self) -> requests.Session:
        """Get or create a requests session with proper configuration."""
        current_time = time.time()
        
        # Create new session if none exists or if current session is too old
        if (
            self._session is None 
            or self._session_created_at is None 
            or current_time - self._session_created_at > self._session_max_age
        ):
            if self._session:
                self._session.close()
            
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "Keep-Grafana-Provider/1.0"
            })
            
            # Configure session with connection pooling and retry strategy
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=0  # We handle retries at a higher level
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
            
            self._session_created_at = current_time
            logger.debug("Created new Grafana HTTP session")
        
        return self._session
    
    def close(self):
        """Close the current session."""
        if self._session:
            self._session.close()
            self._session = None
            self._session_created_at = None
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a GET request with retry logic."""
        session = self.get_session()
        url = f"{self.host.rstrip('/')}/{endpoint.lstrip('/')}"
        kwargs.setdefault('timeout', self.timeout)
        kwargs.setdefault('verify', False)
        
        logger.debug(f"Making GET request to {url}")
        response = session.get(url, **kwargs)
        
        # Log response details
        logger.debug(
            f"GET {url} responded with {response.status_code}",
            extra={"status_code": response.status_code, "url": url}
        )
        
        return response
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a POST request with retry logic."""
        session = self.get_session()
        url = f"{self.host.rstrip('/')}/{endpoint.lstrip('/')}"
        kwargs.setdefault('timeout', self.timeout)
        kwargs.setdefault('verify', False)
        
        logger.debug(f"Making POST request to {url}")
        response = session.post(url, **kwargs)
        
        # Log response details
        logger.debug(
            f"POST {url} responded with {response.status_code}",
            extra={"status_code": response.status_code, "url": url}
        )
        
        return response
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def put(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a PUT request with retry logic."""
        session = self.get_session()
        url = f"{self.host.rstrip('/')}/{endpoint.lstrip('/')}"
        kwargs.setdefault('timeout', self.timeout)
        kwargs.setdefault('verify', False)
        
        logger.debug(f"Making PUT request to {url}")
        response = session.put(url, **kwargs)
        
        # Log response details
        logger.debug(
            f"PUT {url} responded with {response.status_code}",
            extra={"status_code": response.status_code, "url": url}
        )
        
        return response
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def delete(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a DELETE request with retry logic."""
        session = self.get_session()
        url = f"{self.host.rstrip('/')}/{endpoint.lstrip('/')}"
        kwargs.setdefault('timeout', self.timeout)
        kwargs.setdefault('verify', False)
        
        logger.debug(f"Making DELETE request to {url}")
        response = session.delete(url, **kwargs)
        
        # Log response details
        logger.debug(
            f"DELETE {url} responded with {response.status_code}",
            extra={"status_code": response.status_code, "url": url}
        )
        
        return response
    
    def validate_connection(self) -> bool:
        """Validate the connection to Grafana by checking the health endpoint."""
        try:
            response = self.get("api/health")
            if response.status_code == 200:
                health_data = response.json()
                logger.info(
                    "Grafana connection validated successfully",
                    extra={"version": health_data.get("version", "unknown")}
                )
                return True
            else:
                logger.warning(
                    f"Grafana health check failed with status {response.status_code}",
                    extra={"response": response.text[:500]}
                )
                return False
        except Exception as e:
            logger.error(
                "Failed to validate Grafana connection",
                extra={"error": str(e)}
            )
            return False


@pydantic.dataclasses.dataclass
class GrafanaProviderAuthConfig:
    """
    Grafana authentication configuration.
    """

    token: str = dataclasses.field(
        metadata={
            "required": True,
            "description": "Token",
            "hint": "Grafana Token",
            "sensitive": True,
        },
    )
    host: pydantic.AnyHttpUrl = dataclasses.field(
        metadata={
            "required": True,
            "description": "Grafana host",
            "hint": "e.g. https://keephq.grafana.net",
            "validation": "any_http_url",
        },
    )
    datasource_uid: str = dataclasses.field(
        metadata={
            "required": False,
            "description": "Datasource UID",
            "hint": "Provide if you want to pull topology data",
        },
        default="",
    )


class GrafanaProvider(BaseTopologyProvider, ProviderHealthMixin):
    PROVIDER_DISPLAY_NAME = "Grafana"
    """Pull/Push alerts & Topology map from Grafana."""

    PROVIDER_CATEGORY = ["Monitoring", "Developer Tools"]
    KEEP_GRAFANA_WEBHOOK_INTEGRATION_NAME = "keep-grafana-webhook-integration"
    FINGERPRINT_FIELDS = ["fingerprint"]

    webhook_description = ""
    webhook_template = ""
    webhook_markdown = """If your Grafana is unreachable from Keep, you can use the following webhook url to configure Grafana to send alerts to Keep:

    1. In Grafana, go to the Alerting tab in the Grafana dashboard.
    2. Click on Contact points in the left sidebar and create a new one.
    3. Give it a name and select Webhook as kind of contact point with webhook url as {keep_webhook_api_url}.
    4. Add 'X-API-KEY' as the request header {api_key}.
    5. Save the webhook.
    6. Click on Notification policies in the left sidebar
    7. Click on "New child policy" under the "Default policy"
    8. Remove all matchers until you see the following: "If no matchers are specified, this notification policy will handle all alert instances."
    9. Chose the webhook contact point you have just created under Contact point and click "Save Policy"
    """

    PROVIDER_SCOPES = [
        ProviderScope(
            name="alert.rules:read",
            description="Read Grafana alert rules in a folder and its subfolders.",
            mandatory=True,
            mandatory_for_webhook=False,
            documentation_url="https://grafana.com/docs/grafana/latest/administration/roles-and-permissions/access-control/custom-role-actions-scopes/",
            alias="Rules Reader",
        ),
        ProviderScope(
            name="alert.provisioning:read",
            description="Read all Grafana alert rules, notification policies, etc via provisioning API.",
            mandatory=False,
            mandatory_for_webhook=True,
            documentation_url="https://grafana.com/docs/grafana/latest/administration/roles-and-permissions/access-control/custom-role-actions-scopes/",
            alias="Access to alert rules provisioning API",
        ),
        ProviderScope(
            name="alert.provisioning:write",
            description="Update all Grafana alert rules, notification policies, etc via provisioning API.",
            mandatory=False,
            mandatory_for_webhook=True,
            documentation_url="https://grafana.com/docs/grafana/latest/administration/roles-and-permissions/access-control/custom-role-actions-scopes/",
            alias="Access to alert rules provisioning API",
        ),
    ]

    SEVERITIES_MAP = {
        "critical": AlertSeverity.CRITICAL,
        "high": AlertSeverity.HIGH,
        "warning": AlertSeverity.WARNING,
        "info": AlertSeverity.INFO,
    }

    # https://grafana.com/docs/grafana/latest/alerting/manage-notifications/view-state-health/#alert-instance-state
    STATUS_MAP = {
        "ok": AlertStatus.RESOLVED,
        "resolved": AlertStatus.RESOLVED,
        "normal": AlertStatus.RESOLVED,
        "paused": AlertStatus.SUPPRESSED,
        "alerting": AlertStatus.FIRING,
        "pending": AlertStatus.PENDING,
        "no_data": AlertStatus.PENDING,
    }

    def __init__(
        self, context_manager: ContextManager, provider_id: str, config: ProviderConfig
    ):
        super().__init__(context_manager, provider_id, config)
        self._connection_manager = None

    def dispose(self):
        """
        Dispose the provider.
        """
        if self._connection_manager:
            self._connection_manager.close()
            self._connection_manager = None
        pass
    
    @property
    def connection_manager(self) -> GrafanaConnectionManager:
        """Get or create the connection manager."""
        if self._connection_manager is None:
            self._connection_manager = GrafanaConnectionManager(
                host=str(self.authentication_config.host),
                token=self.authentication_config.token,
                timeout=30
            )
        return self._connection_manager
    
    def validate_connection(self) -> bool:
        """Validate the connection to Grafana."""
        try:
            return self.connection_manager.validate_connection()
        except Exception as e:
            self.logger.error(
                "Failed to validate Grafana connection",
                extra={"error": str(e)}
            )
            return False

    def validate_config(self):
        """
        Validates required configuration for Grafana provider.
        """
        self.authentication_config = GrafanaProviderAuthConfig(
            **self.config.authentication
        )

    def validate_scopes(self) -> dict[str, bool | str]:
        try:
            # First validate basic connection
            if not self.validate_connection():
                validated_scopes = {
                    scope.name: "Failed to connect to Grafana. Please check your host and token."
                    for scope in self.PROVIDER_SCOPES
                }
                return validated_scopes
            
            # Then check permissions
            response = self.connection_manager.get("api/access-control/user/permissions")
            
            if not response.ok:
                error_msg = f"Failed to get permissions (HTTP {response.status_code}). Please check your token."
                validated_scopes = {
                    scope.name: error_msg
                    for scope in self.PROVIDER_SCOPES
                }
                return validated_scopes
            
            permissions_data = response.json()
            
        except requests.exceptions.ConnectionError:
            self.logger.exception("Failed to connect to Grafana")
            validated_scopes = {
                scope.name: "Failed to connect to Grafana. Please check your host."
                for scope in self.PROVIDER_SCOPES
            }
            return validated_scopes
        except Exception:
            self.logger.exception("Failed to get permissions from Grafana")
            validated_scopes = {
                scope.name: "Failed to get permissions. Please check your token."
                for scope in self.PROVIDER_SCOPES
            }
            return validated_scopes
        validated_scopes = {}
        for scope in self.PROVIDER_SCOPES:
            if scope.name in permissions_data:
                validated_scopes[scope.name] = True
            else:
                validated_scopes[scope.name] = "Missing scope"
        return validated_scopes

    def get_provider_metadata(self) -> dict:
        version = self._get_grafana_version()
        return {
            "version": version,
        }

    def get_alerts_configuration(self, alert_id: str | None = None):
        try:
            response = self.connection_manager.get("api/v1/provisioning/alert-rules")
            if not response.ok:
                error_data = {}
                try:
                    error_data = response.json()
                except Exception:
                    error_data = {"message": response.text}
                
                self.logger.warning(
                    "Could not get alerts", extra={"response": error_data, "status_code": response.status_code}
                )
                
                if response.status_code == 403:
                    error_data[
                        "message"
                    ] += f"\nYou can test your permissions with \n\tcurl -H 'Authorization: Bearer {{token}}' -X GET '{self.authentication_config.host}/api/access-control/user/permissions' | jq \nDocs: https://grafana.com/docs/grafana/latest/administration/service-accounts/#debug-the-permissions-of-a-service-account-token"
                raise GetAlertException(message=error_data, status_code=response.status_code)
            return response.json()
        except GetAlertException:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to get alerts configuration",
                extra={"error": str(e)}
            )
            raise GetAlertException(
                message={"error": f"Failed to connect to Grafana: {str(e)}"},
                status_code=500
            )

    def deploy_alert(self, alert: dict, alert_id: str | None = None):
        self.logger.info("Deploying alert")
        try:
            response = self.connection_manager.post("api/v1/provisioning/alert-rules", json=alert)

            if not response.ok:
                try:
                    response_json = response.json()
                except Exception:
                    response_json = {"error": response.text}
                
                self.logger.warning(
                    "Could not deploy alert", 
                    extra={"response": response_json, "status_code": response.status_code}
                )
                raise Exception(response_json)

            result = response.json()
            self.logger.info(
                "Alert deployed successfully",
                extra={
                    "response": result,
                    "status": response.status_code,
                },
            )
            return result
        except Exception as e:
            if "Could not deploy alert" not in str(e):
                self.logger.error(
                    "Failed to deploy alert",
                    extra={"error": str(e), "alert": alert}
                )
            raise

    @staticmethod
    def get_alert_schema():
        return GrafanaAlertFormatDescription.schema()

    @staticmethod
    def get_service(alert: dict) -> str:
        """
        Get service from alert.
        """
        labels = alert.get("labels", {})
        return alert.get("service", labels.get("service", "unknown"))

    @staticmethod
    def calculate_fingerprint(alert: dict) -> str:
        """
        Calculate fingerprint for alert.
        """
        labels = alert.get("labels", {})
        fingerprint = labels.get("fingerprint", "")
        if fingerprint:
            logger.debug("Fingerprint provided in alert")
            return fingerprint

        fingerprint_string = None
        if not labels:
            logger.warning(
                "No labels found in alert will use old behaviour",
                extra={
                    "labels": labels,
                },
            )
        else:
            try:
                logger.info(
                    "No fingerprint in alert, calculating fingerprint by labels"
                )
                fingerprint_string = json.dumps(labels)
            except Exception:
                logger.exception(
                    "Failed to calculate fingerprint",
                    extra={
                        "labels": labels,
                    },
                )

        # from some reason, the fingerprint is not provided in the alert + no labels or failed to calculate
        if not fingerprint_string:
            # old behavior
            service = GrafanaProvider.get_service(alert)
            fingerprint_string = alert.get(
                "fingerprint", alert.get("alertname", "") + service
            )

        fingerprint = hashlib.sha256(fingerprint_string.encode()).hexdigest()
        return fingerprint

    @staticmethod
    def _format_alert(
        event: dict, provider_instance: "BaseProvider" = None
    ) -> AlertDto:
        # Check if this is a legacy alert based on structure
        if "evalMatches" in event:
            return GrafanaProvider._format_legacy_alert(event)

        alerts = event.get("alerts", [])

        logger.info("Formatting Grafana alerts", extra={"num_of_alerts": len(alerts)})

        formatted_alerts = []
        for alert in alerts:
            labels = alert.get("labels", {})
            # map status and severity to Keep format:
            status = GrafanaProvider.STATUS_MAP.get(
                event.get("status"), AlertStatus.FIRING
            )
            severity = GrafanaProvider.SEVERITIES_MAP.get(
                labels.get("severity"), AlertSeverity.INFO
            )
            fingerprint = GrafanaProvider.calculate_fingerprint(alert)
            environment = labels.get(
                "deployment_environment", labels.get("environment", "unknown")
            )

            extra = {}

            annotations = alert.get("annotations", {})
            if annotations:
                extra["annotations"] = annotations
            values = alert.get("values", {})
            if values:
                extra["values"] = values

            url = alert.get("generatorURL", None)
            image_url = alert.get("imageURL", None)
            dashboard_url = alert.get("dashboardURL", None)
            panel_url = alert.get("panelURL", None)

            description = alert.get("annotations", {}).get("description") or alert.get(
                "annotations", {}
            ).get("summary", "")

            valueString = alert.get("valueString")

            alert_dto = AlertDto(
                id=alert.get("fingerprint"),
                fingerprint=fingerprint,
                name=event.get("title"),
                status=status,
                severity=severity,
                environment=environment,
                lastReceived=datetime.datetime.now(
                    tz=datetime.timezone.utc
                ).isoformat(),
                description=description,
                source=["grafana"],
                labels=labels,
                url=url or None,
                imageUrl=image_url or None,
                dashboardUrl=dashboard_url or None,
                panelUrl=panel_url or None,
                valueString=valueString,
                **extra,  # add annotations and values
            )
            # enrich extra payload with labels
            for label in labels:
                if getattr(alert_dto, label, None) is None:
                    setattr(alert_dto, label, labels[label])
            formatted_alerts.append(alert_dto)
        return formatted_alerts

    @staticmethod
    def _format_legacy_alert(event: dict) -> AlertDto:
        # Legacy alerts have a different structure
        status = (
            AlertStatus.FIRING
            if event.get("state") == "alerting"
            else AlertStatus.RESOLVED
        )
        severity = GrafanaProvider.SEVERITIES_MAP.get("critical", AlertSeverity.INFO)

        alert_dto = AlertDto(
            id=str(event.get("ruleId", "")),
            fingerprint=str(event.get("ruleId", "")),
            name=event.get("ruleName", ""),
            status=status,
            severity=severity,
            lastReceived=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            description=event.get("message", ""),
            source=["grafana"],
            labels={
                "metric": event.get("metric", ""),
                "ruleId": str(event.get("ruleId", "")),
                "ruleName": event.get("ruleName", ""),
                "ruleUrl": event.get("ruleUrl", ""),
                "state": event.get("state", ""),
            },
        )
        return [alert_dto]

    def _get_grafana_version(self) -> str:
        """Get the Grafana version."""
        try:
            resp = self.connection_manager.get("api/health")

            if resp.ok:
                health_data = resp.json()
                version = health_data.get("version", "unknown")
                self.logger.debug(f"Retrieved Grafana version: {version}")
                return version
            else:
                self.logger.warning(
                    f"Failed to get Grafana version: {resp.status_code}",
                    extra={"response": resp.text[:200]}
                )
                return "unknown"
        except Exception as e:
            self.logger.error(f"Error getting Grafana version: {str(e)}")
            return "unknown"

    def setup_webhook(
        self, tenant_id: str, keep_api_url: str, api_key: str, setup_alerts: bool = True
    ):
        self.logger.info("Setting up webhook")
        webhook_name = (
            f"{GrafanaProvider.KEEP_GRAFANA_WEBHOOK_INTEGRATION_NAME}-{tenant_id}"
        )
        headers = {"Authorization": f"Bearer {self.authentication_config.token}"}
        contacts_api = (
            f"{self.authentication_config.host}/api/v1/provisioning/contact-points"
        )
        try:
            self.logger.info("Getting contact points")
            all_contact_points = requests.get(
                contacts_api, verify=False, headers=headers
            )
            all_contact_points.raise_for_status()
            all_contact_points = all_contact_points.json()
        except Exception:
            self.logger.exception("Failed to get contact points")
            raise
        # check if webhook already exists
        webhook_exists = [
            webhook_exists
            for webhook_exists in all_contact_points
            if webhook_exists.get("name") == webhook_name
            or webhook_exists.get("uid") == webhook_name
        ]
        # grafana version lesser then 9.4.7 do not send their authentication correctly
        # therefor we need to add the api_key as a query param instead of the normal digest token
        self.logger.info("Getting Grafana version")
        try:
            grafana_version = self._get_grafana_version()
        except Exception:
            self.logger.exception("Failed to get Grafana version")
            raise
        self.logger.info(f"Grafana version is {grafana_version}")
        # if grafana version is greater then 9.4.7 we can use the digest token
        if Version(grafana_version) > Version("9.4.7"):
            self.logger.info("Installing Grafana version > 9.4.7")
            if webhook_exists:
                webhook = webhook_exists[0]
                webhook["settings"]["url"] = keep_api_url
                webhook["settings"]["authorization_scheme"] = "digest"
                webhook["settings"]["authorization_credentials"] = api_key
                requests.put(
                    f'{contacts_api}/{webhook["uid"]}',
                    verify=False,
                    json=webhook,
                    headers=headers,
                )
                self.logger.info(f'Updated webhook {webhook["uid"]}')
            else:
                self.logger.info('Creating webhook with name "{webhook_name}"')
                webhook = {
                    "name": webhook_name,
                    "type": "webhook",
                    "settings": {
                        "httpMethod": "POST",
                        "url": keep_api_url,
                        "authorization_scheme": "digest",
                        "authorization_credentials": api_key,
                    },
                }
                response = requests.post(
                    contacts_api,
                    verify=False,
                    json=webhook,
                    headers={**headers, "X-Disable-Provenance": "true"},
                )
                if not response.ok:
                    raise Exception(response.json())
                self.logger.info(f"Created webhook {webhook_name}")
        # if grafana version is lesser then 9.4.7 we need to add the api_key as a query param
        else:
            self.logger.info("Installing Grafana version < 9.4.7")
            if webhook_exists:
                webhook = webhook_exists[0]
                webhook["settings"]["url"] = f"{keep_api_url}&api_key={api_key}"
                requests.put(
                    f'{contacts_api}/{webhook["uid"]}',
                    verify=False,
                    json=webhook,
                    headers=headers,
                )
                self.logger.info(f'Updated webhook {webhook["uid"]}')
            else:
                self.logger.info('Creating webhook with name "{webhook_name}"')
                webhook = {
                    "name": webhook_name,
                    "type": "webhook",
                    "settings": {
                        "httpMethod": "POST",
                        "url": f"{keep_api_url}?api_key={api_key}",
                    },
                }
                response = requests.post(
                    contacts_api,
                    verify=False,
                    json=webhook,
                    headers={**headers, "X-Disable-Provenance": "true"},
                )
                if not response.ok:
                    raise Exception(response.json())
                self.logger.info(f"Created webhook {webhook_name}")
        # Finally, we need to update the policies to match the webhook
        if setup_alerts:
            self.logger.info("Setting up alerts")
            policies_api = (
                f"{self.authentication_config.host}/api/v1/provisioning/policies"
            )
            all_policies = requests.get(
                policies_api, verify=False, headers=headers
            ).json()
            policy_exists = any(
                [
                    p
                    for p in all_policies.get("routes", [])
                    if p.get("receiver") == webhook_name
                ]
            )
            if not policy_exists:
                if all_policies["receiver"]:
                    default_policy = {
                        "receiver": all_policies["receiver"],
                        "continue": True,
                    }
                    if not any(
                        [
                            p
                            for p in all_policies.get("routes", [])
                            if p == default_policy
                        ]
                    ):
                        # This is so we won't override the default receiver if customer has one.
                        if "routes" not in all_policies:
                            all_policies["routes"] = []
                        all_policies["routes"].append(
                            {"receiver": all_policies["receiver"], "continue": True}
                        )
                all_policies["routes"].append(
                    {
                        "receiver": webhook_name,
                        "continue": True,
                    }
                )
                requests.put(
                    policies_api,
                    verify=False,
                    json=all_policies,
                    headers={**headers, "X-Disable-Provenance": "true"},
                )
                self.logger.info("Updated policices to match alerts to webhook")
            else:
                self.logger.info("Policies already match alerts to webhook")

        # After setting up unified alerting, check and setup legacy alerting if enabled
        try:
            self.logger.info("Checking legacy alerting")
            if self._is_legacy_alerting_enabled():
                self.logger.info("Legacy alerting is enabled")
                self._setup_legacy_alerting_webhook(
                    webhook_name, keep_api_url, api_key, setup_alerts
                )
                self.logger.info("Legacy alerting setup successful")

        except Exception:
            self.logger.warning(
                "Failed to check or setup legacy alerting", exc_info=True
            )

        self.logger.info("Webhook successfuly setup")

    def delete_webhook(self, tenant_id: str) -> dict:
        """
        Delete webhook configuration from both unified and legacy alerting.
        
        Args:
            tenant_id: The tenant ID used to identify the webhook
            
        Returns:
            dict: Status report of deletion operations
        """
        webhook_name = (
            f"{GrafanaProvider.KEEP_GRAFANA_WEBHOOK_INTEGRATION_NAME}-{tenant_id}"
        )
        
        deletion_status = {
            "unified_alerting": {"deleted": False, "error": None},
            "legacy_alerting": {"deleted": False, "error": None},
            "webhook_name": webhook_name
        }
        
        self.logger.info(f"Starting webhook deletion for webhook: {webhook_name}")
        
        try:
            # First validate connection
            if not self.validate_connection():
                error_msg = "Failed to connect to Grafana. Cannot delete webhook."
                deletion_status["unified_alerting"]["error"] = error_msg
                deletion_status["legacy_alerting"]["error"] = error_msg
                return deletion_status
            
            # Delete from unified alerting
            try:
                self._delete_unified_alerting_webhook(webhook_name, deletion_status)
            except Exception as e:
                self.logger.error(
                    "Failed to delete unified alerting webhook",
                    extra={"error": str(e), "webhook_name": webhook_name}
                )
                deletion_status["unified_alerting"]["error"] = str(e)
            
            # Delete from legacy alerting (if enabled)
            try:
                if self._is_legacy_alerting_enabled():
                    self._delete_legacy_alerting_webhook(webhook_name, deletion_status)
                else:
                    self.logger.info("Legacy alerting not enabled, skipping legacy webhook deletion")
                    deletion_status["legacy_alerting"]["error"] = "Legacy alerting not enabled"
            except Exception as e:
                self.logger.error(
                    "Failed to delete legacy alerting webhook",
                    extra={"error": str(e), "webhook_name": webhook_name}
                )
                deletion_status["legacy_alerting"]["error"] = str(e)
            
            # Summary logging
            if deletion_status["unified_alerting"]["deleted"] or deletion_status["legacy_alerting"]["deleted"]:
                self.logger.info(
                    "Webhook deletion completed",
                    extra={"status": deletion_status}
                )
            else:
                self.logger.warning(
                    "No webhooks were deleted",
                    extra={"status": deletion_status}
                )
            
            return deletion_status
            
        except Exception as e:
            self.logger.error(
                "Failed to delete webhook",
                extra={"error": str(e), "webhook_name": webhook_name}
            )
            deletion_status["unified_alerting"]["error"] = str(e)
            deletion_status["legacy_alerting"]["error"] = str(e)
            return deletion_status
    
    def _delete_unified_alerting_webhook(self, webhook_name: str, deletion_status: dict):
        """Delete webhook from unified alerting system."""
        self.logger.info("Deleting unified alerting webhook")
        
        # Get all contact points
        response = self.connection_manager.get("api/v1/provisioning/contact-points")
        if not response.ok:
            raise Exception(f"Failed to get contact points: {response.status_code} - {response.text}")
        
        contact_points = response.json()
        webhook_to_delete = None
        
        # Find the webhook
        for contact_point in contact_points:
            if contact_point.get("name") == webhook_name or contact_point.get("uid") == webhook_name:
                webhook_to_delete = contact_point
                break
        
        if not webhook_to_delete:
            self.logger.info(f"Unified alerting webhook '{webhook_name}' not found")
            deletion_status["unified_alerting"]["error"] = "Webhook not found"
            return
        
        # Delete the contact point
        webhook_uid = webhook_to_delete["uid"]
        delete_response = self.connection_manager.delete(f"api/v1/provisioning/contact-points/{webhook_uid}")
        
        if delete_response.ok:
            self.logger.info(f"Successfully deleted unified alerting contact point: {webhook_uid}")
            deletion_status["unified_alerting"]["deleted"] = True
            
            # Remove from notification policies
            try:
                self._remove_from_notification_policies(webhook_name)
            except Exception as e:
                self.logger.warning(
                    "Failed to remove webhook from notification policies",
                    extra={"error": str(e), "webhook_name": webhook_name}
                )
        else:
            error_msg = f"Failed to delete contact point: {delete_response.status_code} - {delete_response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def _delete_legacy_alerting_webhook(self, webhook_name: str, deletion_status: dict):
        """Delete webhook from legacy alerting system."""
        self.logger.info("Deleting legacy alerting webhook")
        
        # Get all notification channels
        response = self.connection_manager.get("api/alert-notifications")
        if not response.ok:
            if response.status_code == 404:
                self.logger.info("Legacy alerting API not available")
                deletion_status["legacy_alerting"]["error"] = "Legacy alerting not available"
                return
            raise Exception(f"Failed to get notification channels: {response.status_code} - {response.text}")
        
        notification_channels = response.json()
        channel_to_delete = None
        
        # Find the notification channel
        for channel in notification_channels:
            if channel.get("name") == webhook_name:
                channel_to_delete = channel
                break
        
        if not channel_to_delete:
            self.logger.info(f"Legacy alerting webhook '{webhook_name}' not found")
            deletion_status["legacy_alerting"]["error"] = "Webhook not found"
            return
        
        # Delete the notification channel
        channel_uid = channel_to_delete["uid"]
        delete_response = self.connection_manager.delete(f"api/alert-notifications/{channel_uid}")
        
        if delete_response.ok:
            self.logger.info(f"Successfully deleted legacy alerting notification channel: {channel_uid}")
            deletion_status["legacy_alerting"]["deleted"] = True
            
            # Remove from alerts (this is more complex and might require updating individual dashboards)
            try:
                self._remove_from_legacy_alerts(channel_uid)
            except Exception as e:
                self.logger.warning(
                    "Failed to remove notification channel from alerts",
                    extra={"error": str(e), "channel_uid": channel_uid}
                )
        else:
            error_msg = f"Failed to delete notification channel: {delete_response.status_code} - {delete_response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def _remove_from_notification_policies(self, webhook_name: str):
        """Remove webhook from notification policies."""
        self.logger.info("Removing webhook from notification policies")
        
        # Get current policies
        response = self.connection_manager.get("api/v1/provisioning/policies")
        if not response.ok:
            raise Exception(f"Failed to get notification policies: {response.status_code} - {response.text}")
        
        policies = response.json()
        modified = False
        
        # Remove webhook from routes
        if "routes" in policies:
            original_routes = policies["routes"][:]
            policies["routes"] = [
                route for route in policies["routes"]
                if route.get("receiver") != webhook_name
            ]
            if len(policies["routes"]) != len(original_routes):
                modified = True
                self.logger.info(f"Removed {len(original_routes) - len(policies['routes'])} routes using webhook")
        
        # Update policies if modified
        if modified:
            update_response = self.connection_manager.put(
                "api/v1/provisioning/policies",
                json=policies,
                headers={"X-Disable-Provenance": "true"}
            )
            if not update_response.ok:
                raise Exception(f"Failed to update notification policies: {update_response.status_code} - {update_response.text}")
            self.logger.info("Successfully updated notification policies")
        else:
            self.logger.info("No notification policies needed updating")
    
    def _remove_from_legacy_alerts(self, channel_uid: str):
        """Remove notification channel from legacy alerts."""
        self.logger.info("Removing notification channel from legacy alerts")
        
        # Get all alerts
        try:
            all_alerts = self._get_all_alerts("api/alerts", {})
            updated_count = 0
            
            for alert in all_alerts:
                dashboard_uid = alert.get("dashboardUid")
                panel_id = alert.get("panelId")
                
                if dashboard_uid and panel_id:
                    try:
                        if self._remove_notification_from_dashboard_alert(dashboard_uid, panel_id, channel_uid):
                            updated_count += 1
                        time.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to update alert in dashboard {dashboard_uid}, panel {panel_id}",
                            extra={"error": str(e)}
                        )
            
            self.logger.info(f"Updated {updated_count} legacy alerts to remove notification channel")
            
        except Exception as e:
            self.logger.warning(
                "Failed to remove notification channel from legacy alerts",
                extra={"error": str(e)}
            )
    
    def _remove_notification_from_dashboard_alert(self, dashboard_uid: str, panel_id: int, notification_uid: str) -> bool:
        """Remove notification from a single dashboard alert."""
        try:
            # Get the dashboard
            response = self.connection_manager.get(f"api/dashboards/uid/{dashboard_uid}")
            if not response.ok:
                return False
            
            dashboard = response.json()["dashboard"]
            updated = False
            
            # Find the panel and update its alert
            for panel in dashboard.get("panels", []):
                if panel.get("id") == panel_id and "alert" in panel:
                    if "notifications" in panel["alert"]:
                        original_notifications = panel["alert"]["notifications"][:]
                        panel["alert"]["notifications"] = [
                            notif for notif in panel["alert"]["notifications"]
                            if notif.get("uid") != notification_uid
                        ]
                        if len(panel["alert"]["notifications"]) != len(original_notifications):
                            updated = True
            
            if updated:
                # Update the dashboard
                update_response = self.connection_manager.post(
                    "api/dashboards/db",
                    json={"dashboard": dashboard, "overwrite": True}
                )
                return update_response.ok
            
            return False
            
        except Exception:
            return False

    def get_manual_webhook_deletion_instructions(self, tenant_id: str) -> str:
        """
        Generate manual webhook deletion instructions for cases where automatic deletion fails.
        
        Args:
            tenant_id: The tenant ID used to identify the webhook
            
        Returns:
            str: Formatted instructions for manual webhook deletion
        """
        webhook_name = f"{GrafanaProvider.KEEP_GRAFANA_WEBHOOK_INTEGRATION_NAME}-{tenant_id}"
        host = str(self.authentication_config.host)
        
        instructions = f"""If automatic webhook deletion failed, you can manually remove the webhook configuration using the following steps:

    Webhook Name: {webhook_name}
    Grafana Host: {host}

    Method 1 - Using Grafana UI (Unified Alerting):
    1. Log into your Grafana instance at {host}
    2. Navigate to Alerting > Contact points
    3. Look for a contact point named {webhook_name}
    4. Click the Delete button (trash icon) next to the contact point
    5. Confirm the deletion
    6. Navigate to Alerting > Notification policies
    7. Remove any policies that reference the deleted contact point

    Method 2 - Using Grafana UI (Legacy Alerting):
    1. Log into your Grafana instance at {host}
    2. Navigate to Alerting > Notification channels
    3. Look for a notification channel named {webhook_name}
    4. Click the Delete button next to the notification channel
    5. Confirm the deletion
    6. Review your dashboard alerts and remove references to the deleted notification channel

    Method 3 - Using Grafana API:
    1. Get all contact points: curl -H "Authorization: Bearer YOUR_TOKEN" -X GET "{host}/api/v1/provisioning/contact-points"
    2. Find the webhook contact point UID from the response (look for name: {webhook_name})
    3. Delete the contact point: curl -H "Authorization: Bearer YOUR_TOKEN" -X DELETE "{host}/api/v1/provisioning/contact-points/CONTACT_POINT_UID"
    4. For legacy alerting, use: curl -H "Authorization: Bearer YOUR_TOKEN" -X GET "{host}/api/alert-notifications" and then curl -H "Authorization: Bearer YOUR_TOKEN" -X DELETE "{host}/api/alert-notifications/NOTIFICATION_CHANNEL_UID"
    
    Note: Replace YOUR_TOKEN with your actual Grafana service account token, and CONTACT_POINT_UID/NOTIFICATION_CHANNEL_UID with the actual UIDs found in step 2.
    """
        return instructions

    def _get_all_alerts(self, alerts_api: str, headers: dict) -> list:
        """Helper function to get all alerts with proper pagination"""
        all_alerts = []
        page = 0
        page_size = 1000  # Grafana's recommended limit

        try:
            while True:
                params = {
                    "dashboardId": None,
                    "panelId": None,
                    "limit": page_size,
                    "startAt": page * page_size,
                }

                self.logger.debug(
                    f"Fetching alerts page {page + 1}", extra={"params": params}
                )

                response = requests.get(
                    alerts_api, params=params, verify=False, headers=headers, timeout=30
                )
                response.raise_for_status()

                page_alerts = response.json()
                if not page_alerts:  # No more alerts to fetch
                    break

                all_alerts.extend(page_alerts)

                # If we got fewer alerts than the page size, we've reached the end
                if len(page_alerts) < page_size:
                    break

                page += 1
                time.sleep(0.2)  # Add delay to avoid rate limiting

            self.logger.info(f"Successfully fetched {len(all_alerts)} alerts")
            return all_alerts

        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to fetch alerts", extra={"error": str(e)})
            raise

    def _is_legacy_alerting_enabled(self) -> bool:
        """Check if legacy alerting is enabled by trying to access legacy endpoints"""
        try:
            headers = {"Authorization": f"Bearer {self.authentication_config.token}"}
            notification_api = (
                f"{self.authentication_config.host}/api/alert-notifications"
            )
            response = requests.get(notification_api, verify=False, headers=headers)
            # If we get a 404, legacy alerting is disabled
            # If we get a 200, legacy alerting is enabled
            # If we get a 401/403, we don't have permissions
            return response.status_code == 200
        except Exception:
            self.logger.warning("Failed to check legacy alerting status", exc_info=True)
            return False

    def _update_dashboard_alert(
        self, dashboard_uid: str, panel_id: int, notification_uid: str, headers: dict
    ) -> bool:
        """Helper function to update a single dashboard alert"""
        try:
            # Get the dashboard
            dashboard_api = (
                f"{self.authentication_config.host}/api/dashboards/uid/{dashboard_uid}"
            )
            dashboard_response = requests.get(
                dashboard_api, verify=False, headers=headers, timeout=30
            )
            dashboard_response.raise_for_status()

            dashboard = dashboard_response.json()["dashboard"]
            updated = False

            # Find the panel and update its alert
            for panel in dashboard.get("panels", []):
                if panel.get("id") == panel_id and "alert" in panel:
                    if "notifications" not in panel["alert"]:
                        panel["alert"]["notifications"] = []
                    # Check if notification already exists
                    if not any(
                        notif.get("uid") == notification_uid
                        for notif in panel["alert"]["notifications"]
                    ):
                        panel["alert"]["notifications"].append(
                            {"uid": notification_uid}
                        )
                        updated = True

            if updated:
                # Update the dashboard
                update_dashboard_api = (
                    f"{self.authentication_config.host}/api/dashboards/db"
                )
                update_response = requests.post(
                    update_dashboard_api,
                    verify=False,
                    json={"dashboard": dashboard, "overwrite": True},
                    headers=headers,
                    timeout=30,
                )
                update_response.raise_for_status()
                return True

            return False

        except requests.exceptions.RequestException as e:
            self.logger.warning(
                f"Failed to update dashboard {dashboard_uid}", extra={"error": str(e)}
            )
            return False

    def _setup_legacy_alerting_webhook(
        self,
        webhook_name: str,
        keep_api_url: str,
        api_key: str,
        setup_alerts: bool = True,
    ):
        """Setup webhook for legacy alerting"""
        self.logger.info("Setting up legacy alerting notification channel")
        headers = {"Authorization": f"Bearer {self.authentication_config.token}"}

        try:
            # Create legacy notification channel
            notification_api = (
                f"{self.authentication_config.host}/api/alert-notifications"
            )
            self.logger.debug(f"Using notification API endpoint: {notification_api}")

            notification = {
                "name": webhook_name,
                "type": "webhook",
                "isDefault": False,
                "sendReminder": False,
                "settings": {
                    "url": keep_api_url,
                    "httpMethod": "POST",
                    "username": "keep",
                    "password": api_key,
                },
            }
            self.logger.debug(f"Prepared notification config: {notification}")

            # Check if notification channel exists
            self.logger.info("Checking for existing notification channels")
            existing_channels = requests.get(
                notification_api, verify=False, headers=headers
            ).json()
            self.logger.debug(f"Found {len(existing_channels)} existing channels")

            channel_exists = any(
                channel
                for channel in existing_channels
                if channel.get("name") == webhook_name
            )

            if not channel_exists:
                self.logger.info(f"Creating new notification channel '{webhook_name}'")
                response = requests.post(
                    notification_api, verify=False, json=notification, headers=headers
                )
                if not response.ok:
                    error_msg = response.json()
                    self.logger.error(
                        f"Failed to create notification channel: {error_msg}"
                    )
                    raise Exception(error_msg)

                notification_uid = response.json().get("uid")
                self.logger.info(
                    f"Created legacy notification channel with UID: {notification_uid}"
                )
            else:
                self.logger.info(
                    f"Legacy notification channel '{webhook_name}' already exists"
                )
                notification_uid = next(
                    channel["uid"]
                    for channel in existing_channels
                    if channel.get("name") == webhook_name
                )
                self.logger.debug(
                    f"Using existing notification channel UID: {notification_uid}"
                )

            if setup_alerts:
                alerts_api = f"{self.authentication_config.host}/api/alerts"
                self.logger.info("Starting alert setup process")

                # Get all alerts using the helper function
                self.logger.info("Fetching all alerts")
                all_alerts = self._get_all_alerts(alerts_api, headers)
                self.logger.info(f"Found {len(all_alerts)} alerts to process")

                updated_count = 0
                for alert in all_alerts:
                    dashboard_uid = alert.get("dashboardUid")
                    panel_id = alert.get("panelId")

                    if dashboard_uid and panel_id:
                        self.logger.debug(
                            f"Processing alert - Dashboard: {dashboard_uid}, Panel: {panel_id}"
                        )
                        if self._update_dashboard_alert(
                            dashboard_uid, panel_id, notification_uid, headers
                        ):
                            updated_count += 1
                            self.logger.debug(
                                f"Successfully updated alert {updated_count}"
                            )
                        # Add delay to avoid rate limiting
                        time.sleep(0.1)

                self.logger.info(
                    f"Completed alert updates - Updated {updated_count} alerts with notification channel"
                )

        except Exception as e:
            self.logger.exception(f"Failed to setup legacy alerting: {str(e)}")
            raise

    def __extract_rules(self, alerts: dict, source: list) -> list[AlertDto]:
        alert_ids = []
        alert_dtos = []
        for group in alerts.get("data", {}).get("groups", []):
            for rule in group.get("rules", []):
                for alert in rule.get("alerts", []):
                    alert_id = rule.get(
                        "id", rule.get("name", "").replace(" ", "_").lower()
                    )

                    if alert_id in alert_ids:
                        # de duplicate alerts
                        continue

                    description = alert.get("annotations", {}).pop(
                        "description", None
                    ) or alert.get("annotations", {}).get("summary", rule.get("name"))

                    labels = {k.lower(): v for k, v in alert.get("labels", {}).items()}
                    annotations = {
                        k.lower(): v for k, v in alert.get("annotations", {}).items()
                    }
                    try:
                        status = alert.get("state", rule.get("state"))
                        status = GrafanaProvider.STATUS_MAP.get(
                            status, AlertStatus.FIRING
                        )
                        alert_dto = AlertDto(
                            id=alert_id,
                            name=rule.get("name"),
                            description=description,
                            status=status,
                            lastReceived=alert.get("activeAt"),
                            source=source,
                            **labels,
                            **annotations,
                        )
                        alert_ids.append(alert_id)
                        alert_dtos.append(alert_dto)
                    except Exception:
                        self.logger.warning(
                            "Failed to parse alert",
                            extra={
                                "alert_id": alert_id,
                                "alert_name": rule.get("name"),
                            },
                        )
                        continue
        return alert_dtos

    def _get_alerts_datasource(self) -> list:
        """
        Get raw alerts from all available datasources (Prometheus, Loki, Grafana, Alertmanager).
        Returns a list of raw alert dictionaries, or an empty list if there are errors.
        """
        self.logger.info("Starting to fetch alerts from Grafana datasources")

        headers = {"Authorization": f"Bearer {self.authentication_config.token}"}
        all_alerts = []

        # Step 1: Get all datasources
        try:
            self.logger.info("Fetching list of datasources")
            datasources_url = f"{self.authentication_config.host}/api/datasources"
            datasources_resp = requests.get(
                datasources_url, headers=headers, timeout=5, verify=False
            )

            if datasources_resp.status_code != 200:
                self.logger.error(
                    f"Failed to get datasources: {datasources_resp.status_code}",
                    extra={"response_text": datasources_resp.text[:500]},
                )
                return []

            self.logger.info(
                f"Successfully fetched datasources, got {len(datasources_resp.json())} datasources"
            )
        except Exception as e:
            self.logger.error(f"Error fetching datasources list: {str(e)}")
            return []

        # Step 2: Extract relevant datasources (Prometheus, Loki, Mimir)
        alert_datasources = []
        try:
            for ds in datasources_resp.json():
                if (
                    ds.get("type") in ["prometheus", "loki"]
                    or "mimir" in ds.get("name", "").lower()
                ):
                    alert_datasources.append(
                        {
                            "uid": ds.get("uid"),
                            "name": ds.get("name"),
                            "type": ds.get("type"),
                        }
                    )

            self.logger.info(
                f"Found {len(alert_datasources)} alert-capable datasources"
            )
        except Exception as e:
            self.logger.error(f"Error parsing datasources: {str(e)}")
            return []

        # Step 3: Query alerts from each datasource
        for ds in alert_datasources:
            try:
                # Log the datasource we're about to query
                self.logger.info(
                    f"Querying alerts for datasource: {ds.get('name')}",
                    extra={"datasource": ds},
                )

                # Different endpoint based on datasource type
                if ds.get("type") == "loki":
                    # For Loki, use the Prometheus-compatible alerts endpoint
                    alert_url = f"{self.authentication_config.host}/api/datasources/proxy/uid/{ds.get('uid')}/prometheus/api/v1/alerts"
                else:
                    # For Prometheus/Mimir, use the standard alerts endpoint
                    alert_url = f"{self.authentication_config.host}/api/datasources/proxy/uid/{ds.get('uid')}/api/v1/alerts"

                # Query the alerts endpoint
                self.logger.info(f"Querying {ds.get('name')} alerts at: {alert_url}")
                resp = requests.get(alert_url, headers=headers, timeout=8, verify=False)

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "success" and "alerts" in data.get(
                        "data", {}
                    ):
                        ds_alerts = data["data"]["alerts"]

                        if ds_alerts:  # Only process non-empty alert lists
                            self.logger.info(
                                f"Found {len(ds_alerts)} alerts in {ds.get('name')}"
                            )

                            for alert in ds_alerts:
                                # Tag with source name and type
                                alert["datasource"] = ds.get("name")
                                alert["datasource_type"] = ds.get("type")

                            all_alerts.extend(ds_alerts)
                        else:
                            self.logger.info(f"No alerts found for {ds.get('name')}")
                    else:
                        self.logger.info(
                            f"No alerts data found in response from {ds.get('name')}",
                            extra={
                                "status": data.get("status"),
                                "has_data": "data" in data,
                                "has_alerts": "data" in data
                                and "alerts" in data.get("data", {}),
                            },
                        )
                else:
                    self.logger.warning(
                        f"Failed to get alerts for {ds.get('name')}: {resp.status_code}",
                        extra={"response": resp.text[:500]},  # Limit response log size
                    )
            except Exception as e:
                self.logger.error(
                    f"Error querying alerts for {ds.get('name')}: {str(e)}",
                    exc_info=True,
                )
                # Continue to the next datasource
                continue

        # Step 4: Process and format the alerts
        formatted_alerts = []
        for alert in all_alerts:
            try:
                # Format the alert using the existing method
                alertname = alert.get(
                    "name",
                    alert.get("alertname", alert.get("labels", {}).get("alertname")),
                )
                if not alertname:
                    logger.warning(
                        "Alert name not found, using default",
                        extra={
                            "alert": alert,
                        },
                    )
                    alertname = "Grafana Alert [Unknown]"
                severity = alert.get(
                    "severity", alert.get("labels", {}).get("severity")
                )
                if not severity:
                    logger.warning(
                        "Alert severity not found, using default",
                        extra={
                            "alert": alert,
                        },
                    )
                    severity = "info"
                severity = GrafanaProvider.SEVERITIES_MAP.get(
                    severity, AlertSeverity.INFO
                )

                status = alert.get("state")
                if not status:
                    logger.warning(
                        "Alert status not found, using default",
                        extra={
                            "alert": alert,
                        },
                    )
                    status = "firing"
                status = GrafanaProvider.STATUS_MAP.get(status, AlertStatus.FIRING)

                labels = alert.get("labels", {})
                # pop severity from labels to avoid duplication
                labels.pop("severity", None)
                annotations = alert.get("annotations", {})

                description = annotations.get("description", annotations.get("summary"))
                try:
                    alert_dto = AlertDto(
                        name=alertname,
                        status=status,
                        severity=severity,
                        source=["grafana"],
                        labels=labels,
                        annotations=annotations,
                        datasource=alert.get("datasource"),
                        datasource_type=alert.get("datasource_type"),
                        value=alert.get("value"),
                    )
                    if description:
                        alert_dto.description = description
                    formatted_alerts.append(alert_dto)
                except Exception:
                    self.logger.exception(
                        "Failed to format datasoruce alert",
                        extra={
                            "alert": alert,
                        },
                    )
                    continue
            except Exception as e:
                self.logger.error(
                    f"Error formatting alert: {str(e)}", extra={"alert": alert}
                )

        self.logger.info(
            f"Total alerts found across all datasources: {len(formatted_alerts)}"
        )
        return formatted_alerts

    def _get_alerts(self) -> list[AlertDto]:
        self.logger.info("Starting to fetch alerts from Grafana")

        # First get alerts from datasources directly
        datasource_alerts = self._get_alerts_datasource()
        self.logger.info(f"Found {len(datasource_alerts)} alerts from datasources")

        # Get Grafana version to determine best approach for history API
        grafana_version = self._get_grafana_version()
        self.logger.info(f"Detected Grafana version: {grafana_version}")

        history_alerts = []

        # Calculate time range (7 days ago to now)
        week_ago = int(
            (datetime.datetime.now() - datetime.timedelta(days=7)).timestamp()
        )
        now = int(datetime.datetime.now().timestamp())
        self.logger.info(
            f"Using time range for alerts: from={week_ago} to={now}",
            extra={"from_timestamp": week_ago, "to_timestamp": now},
        )

        headers = {"Authorization": f"Bearer {self.authentication_config.token}"}

        # First try the general history API (works in older Grafana versions)
        try:
            api_endpoint = f"{self.authentication_config.host}/api/v1/rules/history?from={week_ago}&to={now}&limit=0"
            self.logger.info(f"Querying Grafana history API endpoint: {api_endpoint}")

            response = requests.get(
                api_endpoint, verify=False, headers=headers, timeout=5
            )
            self.logger.info(
                f"Received response from Grafana history API with status code: {response.status_code}"
            )

            if response.ok:
                # Process the response
                events_history = response.json()
                events_data = events_history.get("data", {})

                if events_data and "values" in events_data:
                    events_data_values = events_data.get("values")
                    if events_data_values and len(events_data_values) >= 2:
                        # If we have values, extract the events and timestamps
                        events = events_data_values[1]
                        events_time = events_data_values[0]

                        self.logger.info(f"Found {len(events)} events in history API")

                        for i in range(0, len(events)):
                            event = events[i]
                            try:
                                event_labels = event.get("labels", {})
                                alert_name = event_labels.get("alertname")
                                alert_status = event_labels.get(
                                    "alertstate", event.get("current")
                                )

                                # Map status to Keep format
                                alert_status = GrafanaProvider.STATUS_MAP.get(
                                    alert_status, AlertStatus.FIRING
                                )

                                # Extract other fields
                                alert_severity = event_labels.get("severity")
                                alert_severity = GrafanaProvider.SEVERITIES_MAP.get(
                                    alert_severity, AlertSeverity.INFO
                                )
                                environment = event_labels.get("environment", "unknown")
                                fingerprint = event_labels.get("fingerprint")
                                description = event.get("error", "")
                                rule_id = event.get("ruleUID")
                                condition = event.get("condition")

                                # Convert timestamp
                                timestamp = datetime.datetime.fromtimestamp(
                                    events_time[i] / 1000
                                ).isoformat()

                                # Create AlertDto
                                alert_dto = AlertDto(
                                    id=str(i),
                                    fingerprint=fingerprint,
                                    name=alert_name,
                                    status=alert_status,
                                    severity=alert_severity,
                                    environment=environment,
                                    description=description,
                                    lastReceived=timestamp,
                                    rule_id=rule_id,
                                    condition=condition,
                                    labels=event_labels,
                                    source=["grafana"],
                                )
                                history_alerts.append(alert_dto)
                            except Exception as e:
                                self.logger.error(
                                    f"Error processing event {i+1}",
                                    extra={"event": event, "error": str(e)},
                                )

                self.logger.info(
                    f"Successfully processed {len(history_alerts)} alerts from Grafana history API"
                )
            else:
                # If general API fails with 'ruleUID is required' error in newer Grafana versions
                if "ruleUID is required" in response.text:
                    self.logger.info(
                        "Grafana version requires ruleUID parameter, trying per-rule approach"
                    )

                    # Get all rules first
                    rules_endpoint = (
                        f"{self.authentication_config.host}/api/alerting/rules"
                    )
                    self.logger.info(f"Fetching alert rules from: {rules_endpoint}")

                    rules_response = requests.get(
                        rules_endpoint, verify=False, headers=headers, timeout=5
                    )

                    if rules_response.ok:
                        rules_data = rules_response.json()
                        rule_uids = []

                        # Extract all rule UIDs
                        for group in rules_data.get("data", {}).get("groups", []):
                            for rule in group.get("rules", []):
                                if "uid" in rule:
                                    rule_uids.append(rule["uid"])

                        self.logger.info(f"Found {len(rule_uids)} rule UIDs")

                        # For each rule UID, get its history
                        for rule_uid in rule_uids:
                            rule_history_url = f"{self.authentication_config.host}/api/v1/rules/history?from={week_ago}&to={now}&limit=100&ruleUID={rule_uid}"

                            try:
                                rule_resp = requests.get(
                                    rule_history_url,
                                    verify=False,
                                    headers=headers,
                                    timeout=5,
                                )

                                if rule_resp.ok:
                                    rule_history = rule_resp.json()
                                    rule_data = rule_history.get("data", {})

                                    if rule_data and "values" in rule_data:
                                        rule_values = rule_data.get("values")
                                        if rule_values and len(rule_values) >= 2:
                                            rule_events = rule_values[1]
                                            rule_times = rule_values[0]

                                            self.logger.info(
                                                f"Found {len(rule_events)} events for rule {rule_uid}"
                                            )

                                            for i in range(0, len(rule_events)):
                                                event = rule_events[i]
                                                try:
                                                    event_labels = event.get(
                                                        "labels", {}
                                                    )
                                                    alert_name = event_labels.get(
                                                        "alertname", f"Rule {rule_uid}"
                                                    )
                                                    alert_status = event_labels.get(
                                                        "alertstate",
                                                        event.get("current"),
                                                    )
                                                    alert_status = (
                                                        GrafanaProvider.STATUS_MAP.get(
                                                            alert_status,
                                                            AlertStatus.FIRING,
                                                        )
                                                    )
                                                    alert_severity = event_labels.get(
                                                        "severity"
                                                    )
                                                    alert_severity = GrafanaProvider.SEVERITIES_MAP.get(
                                                        alert_severity,
                                                        AlertSeverity.INFO,
                                                    )
                                                    environment = event_labels.get(
                                                        "environment", "unknown"
                                                    )
                                                    fingerprint = event_labels.get(
                                                        "fingerprint", rule_uid
                                                    )
                                                    description = event.get("error", "")
                                                    condition = event.get("condition")

                                                    # Convert timestamp
                                                    timestamp = (
                                                        datetime.datetime.fromtimestamp(
                                                            rule_times[i] / 1000
                                                        ).isoformat()
                                                    )

                                                    alert_dto = AlertDto(
                                                        id=f"{rule_uid}_{i}",
                                                        fingerprint=fingerprint,
                                                        name=alert_name,
                                                        status=alert_status,
                                                        severity=alert_severity,
                                                        environment=environment,
                                                        description=description,
                                                        lastReceived=timestamp,
                                                        rule_id=rule_uid,
                                                        condition=condition,
                                                        labels=event_labels,
                                                        source=["grafana"],
                                                    )
                                                    history_alerts.append(alert_dto)
                                                except Exception as e:
                                                    self.logger.error(
                                                        f"Error processing event for rule {rule_uid}",
                                                        extra={
                                                            "event": event,
                                                            "error": str(e),
                                                        },
                                                    )
                            except Exception as e:
                                self.logger.error(
                                    f"Error processing history for rule {rule_uid}",
                                    extra={"error": str(e)},
                                )
                    # if response is 404, it means the API is not available
                    elif rules_response.status_code == 404:
                        # if legacy alerting is not enabled, we can assume the API is not available
                        self.logger.error("Grafana history API not available")
                    else:
                        self.logger.error(
                            "Failed to get alerts from Grafana history API",
                            extra={
                                "status_code": response.status_code,
                                "response_text": response.text,
                                "api_endpoint": api_endpoint,
                            },
                        )
                    self.logger.info(
                        f"Processed {len(history_alerts)} alerts from per-rule history API"
                    )
                else:
                    self.logger.error(
                        "Failed to get alerts from Grafana history API",
                        extra={
                            "status_code": response.status_code,
                            "response_text": response.text,
                            "api_endpoint": api_endpoint,
                        },
                    )
        except Exception as e:
            self.logger.error(
                "Error querying Grafana history API", extra={"error": str(e)}
            )

        # Also try to get alerts from Alertmanager
        alertmanager_alerts = []
        try:
            alertmanager_url = f"{self.authentication_config.host}/api/alertmanager/grafana/api/v2/alerts"
            self.logger.info(f"Querying Alertmanager at: {alertmanager_url}")

            am_resp = requests.get(
                alertmanager_url, verify=False, headers=headers, timeout=5
            )

            if am_resp.ok:
                am_alerts_data = am_resp.json()

                if am_alerts_data:
                    self.logger.info(
                        f"Found {len(am_alerts_data)} alerts in Alertmanager"
                    )

                    for i, alert in enumerate(am_alerts_data):
                        try:
                            # Extract alert properties
                            labels = alert.get("labels", {})
                            annotations = alert.get("annotations", {})

                            # Extract alert name
                            alert_name = labels.get("alertname", f"Alert_{i}")

                            # Determine status
                            alert_status = AlertStatus.FIRING
                            if alert.get("status", {}).get("state") == "suppressed":
                                alert_status = AlertStatus.SUPPRESSED
                            elif (
                                alert.get("endsAt")
                                and alert.get("endsAt") != "0001-01-01T00:00:00Z"
                            ):
                                alert_status = AlertStatus.RESOLVED

                            # Extract severity
                            alert_severity = labels.get("severity", "info")
                            alert_severity = GrafanaProvider.SEVERITIES_MAP.get(
                                alert_severity, AlertSeverity.INFO
                            )

                            # Create AlertDto
                            try:
                                alert_dto = AlertDto(
                                    id=alert.get("fingerprint", str(i)),
                                    fingerprint=alert.get("fingerprint"),
                                    name=alert_name,
                                    status=alert_status,
                                    severity=alert_severity,
                                    environment=labels.get("environment", "unknown"),
                                    description=annotations.get(
                                        "description", annotations.get("summary", "")
                                    ),
                                    lastReceived=alert.get("startsAt"),
                                    rule_id=labels.get("ruleId"),
                                    condition="",
                                    labels=labels,
                                    source=["grafana"],
                                )
                                alertmanager_alerts.append(alert_dto)
                            except Exception:
                                self.logger.exception(
                                    f"Error creating AlertDto for Alertmanager alert {i}",
                                    extra={
                                        "alert": alert,
                                    },
                                )
                        except Exception as e:
                            self.logger.error(
                                f"Error processing Alertmanager alert {i}",
                                extra={"alert": alert, "error": str(e)},
                            )
            else:
                self.logger.warning(
                    f"Failed to get alerts from Alertmanager: {am_resp.status_code}"
                )
        except Exception as e:
            self.logger.error("Error querying Alertmanager", extra={"error": str(e)})

        # Combine all alert sources
        all_alerts = datasource_alerts + history_alerts + alertmanager_alerts
        self.logger.info(f"Total alerts found from all sources: {len(all_alerts)}")

        return all_alerts

    @classmethod
    def simulate_alert(cls, **kwargs) -> dict:
        import hashlib
        import json
        import random

        from keep.providers.grafana_provider.alerts_mock import ALERTS

        alert_type = kwargs.get("alert_type")
        if not alert_type:
            alert_type = random.choice(list(ALERTS.keys()))

        to_wrap_with_provider_type = kwargs.get("to_wrap_with_provider_type")

        if "payload" in ALERTS[alert_type]:
            alert_payload = ALERTS[alert_type]["payload"]
        else:
            alert_payload = ALERTS[alert_type]["alerts"][0]
        alert_parameters = ALERTS[alert_type].get("parameters", {})
        alert_renders = ALERTS[alert_type].get("renders", {})
        # Generate random data for parameters
        for parameter, parameter_options in alert_parameters.items():
            if "." in parameter:
                parameter = parameter.split(".")
                if parameter[0] not in alert_payload:
                    alert_payload[parameter[0]] = {}
                alert_payload[parameter[0]][parameter[1]] = random.choice(
                    parameter_options
                )
            else:
                alert_payload[parameter] = random.choice(parameter_options)

        # Apply renders
        for param, choices in alert_renders.items():
            # replace annotations
            # HACK
            param_to_replace = "{{ " + param + " }}"
            alert_payload["annotations"]["summary"] = alert_payload["annotations"][
                "summary"
            ].replace(param_to_replace, random.choice(choices))

        # Implement specific Grafana alert structure here
        # For example:
        alert_payload["state"] = AlertStatus.FIRING.value
        alert_payload["evalMatches"] = [
            {
                "value": random.randint(0, 100),
                "metric": "some_metric",
                "tags": alert_payload.get("labels", {}),
            }
        ]

        # Generate fingerprint
        fingerprint_src = json.dumps(alert_payload, sort_keys=True)
        fingerprint = hashlib.md5(fingerprint_src.encode()).hexdigest()
        alert_payload["fingerprint"] = fingerprint

        final_payload = {
            "alerts": [alert_payload],
            "severity": alert_payload.get("labels", {}).get("severity"),
            "title": alert_type,
        }
        if to_wrap_with_provider_type:
            return {"keep_source_type": "grafana", "event": final_payload}
        return final_payload

    def query_datasource_for_topology(self):
        self.logger.info("Attempting to query datasource for topology data.")
        headers = {
            "Authorization": f"Bearer {self.authentication_config.token}",
            "Content-Type": "application/json",
        }
        json_data = {
            "queries": [
                {
                    "format": "table",
                    "refId": "traces_service_graph_request_total",
                    "expr": "sum by (client, server) (rate(traces_service_graph_request_total[3600s]))",
                    "instant": True,
                    "exemplar": False,
                    "requestId": "service_map_request",
                    "utcOffsetSec": 19800,
                    "interval": "",
                    "legendFormat": "",
                    "datasource": {
                        "uid": self.authentication_config.datasource_uid,
                    },
                    "datasourceId": 1,
                    "intervalMs": 5000,
                    "maxDataPoints": 954,
                },
                {
                    "format": "table",
                    "refId": "traces_service_graph_request_server_seconds_sum",
                    "expr": "sum by (client, server) (rate(traces_service_graph_request_server_seconds_sum[3600s]))",
                    "instant": True,
                    "exemplar": False,
                    "requestId": "service_map_request_avg",
                    "utcOffsetSec": 19800,
                    "interval": "",
                    "legendFormat": "",
                    "datasource": {
                        "uid": self.authentication_config.datasource_uid,
                    },
                    "datasourceId": 1,
                    "intervalMs": 5000,
                    "maxDataPoints": 954,
                },
            ]
        }

        response = self.connection_manager.post("api/ds/query", json=json_data)

        if not response.ok:
            self.logger.error(
                f"Failed to query datasource for topology: {response.status_code}",
                extra={"response": response.text[:500]},
            )
            return []

        topology_data = response.json()
        services = []

        for result in topology_data.get("results", {}).values():
            for frame in result.get("frames", []):
                # Skip frames without data
                if not frame.get("data", {}).get("values"):
                    continue

                # Get the column names and data
                columns = {field["name"]: i for i, field in enumerate(frame.get("schema", {}).get("fields", []))}
                data_values = frame["data"]["values"]

                # Check if we have the required columns
                if "client" not in columns or "server" not in columns:
                    continue

                client_idx = columns["client"]
                server_idx = columns["server"]

                # Extract client-server relationships
                for i in range(len(data_values[client_idx])):
                    client = data_values[client_idx][i]
                    server = data_values[server_idx][i]

                    if client and server:
                        # Add both client and server as services
                        if client not in [s["service"] for s in services]:
                            services.append({"service": client, "dependencies": []})
                        if server not in [s["service"] for s in services]:
                            services.append({"service": server, "dependencies": []})

                        # Add dependency relationship
                        for service in services:
                            if service["service"] == client and server not in service["dependencies"]:
                                service["dependencies"].append(server)

        # Convert to TopologyServiceInDto format
        topology_services = []
        for service_data in services:
            try:
                service = TopologyServiceInDto(
                    source_provider_id=self.provider_id,
                    repository=None,  # Not available from this data
                    tags=None,
                    service=service_data["service"],
                    display_name=service_data["service"],
                    environment="unknown",  # Could be derived from datasource if needed
                    description=f"Service discovered from Grafana traces",
                    team=None,
                    email=None,
                    slack=None,
                    dependencies=service_data["dependencies"]
                )
                topology_services.append(service)
            except Exception as e:
                self.logger.error(f"Failed to create topology service: {e}")
                continue

        self.logger.info(f"Found {len(topology_services)} services in topology")
        return topology_services

    def pull_topology(self) -> list[TopologyServiceInDto]:
        """
        Query Grafana datasource for topology data if datasource_uid is configured.
        """
        if not self.authentication_config.datasource_uid:
            self.logger.info("No datasource UID configured, skipping topology pull")
            return []

        try:
            return self.query_datasource_for_topology()
        except Exception as e:
            self.logger.error(f"Failed to pull topology: {e}")
            return []


if __name__ == "__main__":
    # Test the provider
    import logging

    logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
    context_manager = ContextManager(
        tenant_id="singletenant",
        workflow_id="test",
    )
    # Load environment variables
    import os

    host = os.environ.get("GRAFANA_HOST")
    token = os.environ.get("GRAFANA_TOKEN")

    if host and token:
        config = ProviderConfig(
            description="Grafana Provider",
            authentication={"host": host, "token": token},
        )
        provider = GrafanaProvider(
            context_manager, provider_id="grafana-test", config=config
        )
        result = provider.validate_scopes()
        print(result)
        results = provider._get_alerts()
        print(results)
    else:
        print("Please set GRAFANA_HOST and GRAFANA_TOKEN environment variables")