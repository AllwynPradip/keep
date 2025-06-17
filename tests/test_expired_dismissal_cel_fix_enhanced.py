import datetime
import json
import time
from datetime import timezone, timedelta

import pytest
from freezegun import freeze_time
from keep.api.bl.enrichments_bl import EnrichmentsBl
from keep.api.core.alerts import query_last_alerts
from keep.api.core.db import cleanup_expired_dismissals, get_session
from keep.api.models.action_type import ActionType
from keep.api.models.alert import AlertDto, AlertStatus, AlertSeverity
from keep.api.models.query import QueryDto
from keep.api.utils.enrichment_helpers import convert_db_alerts_to_dto_alerts
from keep.rulesengine.rulesengine import RulesEngine
from tests.fixtures.client import client, setup_api_key, test_app


@pytest.mark.parametrize("test_app", ["NO_AUTH"], indirect=True)
def test_time_travel_dismissal_expiration(
    db_session, test_app, create_alert, caplog
):
    """Test actual time passing scenario using freezegun - most realistic test."""
    
    # Start at a specific time
    start_time = datetime.datetime(2025, 6, 17, 10, 0, 0, tzinfo=timezone.utc)
    
    with freeze_time(start_time) as frozen_time:
        # Create an alert at 10:00 AM
        alert = create_alert(
            "time-travel-alert",
            AlertStatus.FIRING,
            start_time,
            {
                "name": "Time Travel Test Alert",
                "severity": "critical",
                "service": "time-service",
            },
        )
        
        # Dismiss the alert until 10:30 AM (30 minutes later)
        dismiss_until_time = start_time + timedelta(minutes=30)
        dismiss_until_str = dismiss_until_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        caplog.clear()
        
        enrichment_bl = EnrichmentsBl("keep", db=db_session)
        enrichment_bl.enrich_entity(
            fingerprint=alert.fingerprint,
            enrichments={
                "dismissed": True,
                "dismissedUntil": dismiss_until_str,
                "note": "Dismissed for 30 minutes"
            },
            action_type=ActionType.GENERIC_ENRICH,
            action_callee="test_user",
            action_description="Time travel dismissal test"
        )
        
        # At 10:00 AM - alert should be dismissed
        print(f"\n=== Time: {frozen_time.time_to_freeze} (Alert dismissed until {dismiss_until_time}) ===")
        
        # Test CEL filter for dismissed == true (should find the alert)
        db_alerts, total_count = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == true", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        assert len(alerts_dto) == 1
        assert alerts_dto[0].fingerprint == alert.fingerprint
        assert alerts_dto[0].dismissed is True
        print(f"✓ At 10:00 AM: Alert correctly appears in dismissed == true filter")
        
        # Test CEL filter for dismissed == false (should NOT find the alert)
        db_alerts, total_count = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        assert len(alerts_dto) == 0
        print(f"✓ At 10:00 AM: Alert correctly does NOT appear in dismissed == false filter")
        
        # Travel to 10:15 AM - alert should still be dismissed
        frozen_time.tick(timedelta(minutes=15))
        print(f"\n=== Time: {frozen_time.time_to_freeze} (Still within dismissal period) ===")
        
        caplog.clear()
        
        # Test dismissed == true (should still find the alert)
        db_alerts, total_count = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == true", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        assert len(alerts_dto) == 1
        assert alerts_dto[0].dismissed is True
        print(f"✓ At 10:15 AM: Alert still correctly dismissed")
        
        # Check that cleanup ran but found no expired dismissals
        assert "No expired dismissals found to clean up" in caplog.text
        print(f"✓ At 10:15 AM: Cleanup correctly identified no expired dismissals")
        
        # Travel to 10:45 AM - PAST the dismissal expiration time
        frozen_time.tick(timedelta(minutes=30))  # Now at 10:45 AM, dismissed until 10:30 AM
        print(f"\n=== Time: {frozen_time.time_to_freeze} (PAST dismissal expiration!) ===")
        
        caplog.clear()
        
        # Now test dismissed == false - the cleanup should run and find the alert
        db_alerts, total_count = query_last_alerts(
            tenant_id="keep", 
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        # This is the key test - after expiration, alert should appear in dismissed == false
        assert len(alerts_dto) == 1
        assert alerts_dto[0].fingerprint == alert.fingerprint
        assert alerts_dto[0].dismissed is False
        print(f"✅ At 10:45 AM: Alert correctly appears in dismissed == false filter after expiration!")
        
        # Verify cleanup logs show the dismissal was updated
        assert "Starting cleanup of expired dismissals" in caplog.text
        assert "Updating expired dismissal for alert" in caplog.text 
        assert "Successfully updated expired dismissal" in caplog.text
        print(f"✓ At 10:45 AM: Cleanup logs confirm dismissal was properly updated")
        
        # Test dismissed == true - should NOT find the expired alert
        db_alerts, total_count = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == true", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        assert len(alerts_dto) == 0
        print(f"✓ At 10:45 AM: Alert correctly does NOT appear in dismissed == true filter after expiration")
        
        print(f"\n🎉 Time travel test completed successfully!")


@pytest.mark.parametrize("test_app", ["NO_AUTH"], indirect=True)
def test_multiple_alerts_mixed_expiration_times(
    db_session, test_app, create_alert, caplog
):
    """Test multiple alerts with different expiration times using freezegun."""
    
    start_time = datetime.datetime(2025, 6, 17, 14, 0, 0, tzinfo=timezone.utc)
    
    with freeze_time(start_time) as frozen_time:
        # Create 3 alerts with different dismissal periods
        alert1 = create_alert(
            "alert-expires-in-10min",
            AlertStatus.FIRING,
            start_time,
            {"name": "Alert 1 - Expires in 10min", "severity": "critical"},
        )
        
        alert2 = create_alert(
            "alert-expires-in-30min",
            AlertStatus.FIRING,
            start_time,
            {"name": "Alert 2 - Expires in 30min", "severity": "warning"},
        )
        
        alert3 = create_alert(
            "alert-never-expires",
            AlertStatus.FIRING,
            start_time,
            {"name": "Alert 3 - Never expires", "severity": "info"},
        )
        
        enrichment_bl = EnrichmentsBl("keep", db=db_session)
        
        # Dismiss alert1 until 14:10 (10 minutes)
        dismiss_time_1 = start_time + timedelta(minutes=10)
        enrichment_bl.enrich_entity(
            fingerprint=alert1.fingerprint,
            enrichments={
                "dismissed": True,
                "dismissedUntil": dismiss_time_1.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "note": "Dismissed for 10 minutes"
            },
            action_type=ActionType.GENERIC_ENRICH,
            action_callee="test_user",
            action_description="Short dismissal"
        )
        
        # Dismiss alert2 until 14:30 (30 minutes)
        dismiss_time_2 = start_time + timedelta(minutes=30)
        enrichment_bl.enrich_entity(
            fingerprint=alert2.fingerprint,
            enrichments={
                "dismissed": True,
                "dismissedUntil": dismiss_time_2.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "note": "Dismissed for 30 minutes"
            },
            action_type=ActionType.GENERIC_ENRICH,
            action_callee="test_user",
            action_description="Medium dismissal"
        )
        
        # Dismiss alert3 forever
        enrichment_bl.enrich_entity(
            fingerprint=alert3.fingerprint,
            enrichments={
                "dismissed": True,
                "dismissedUntil": "forever",
                "note": "Dismissed forever"
            },
            action_type=ActionType.GENERIC_ENRICH,
            action_callee="test_user",
            action_description="Forever dismissal"
        )
        
        print(f"\n=== Time: {frozen_time.time_to_freeze} - All alerts dismissed ===")
        
        # At 14:00 - all alerts should be dismissed
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == true", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        assert len(alerts_dto) == 3
        print(f"✓ All 3 alerts correctly dismissed initially")
        
        # No alerts should be in not-dismissed
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        assert len(alerts_dto) == 0
        print(f"✓ No alerts in non-dismissed filter initially")
        
        # Travel to 14:15 - alert1 should have expired, others still dismissed
        frozen_time.tick(timedelta(minutes=15))
        print(f"\n=== Time: {frozen_time.time_to_freeze} - Alert1 should have expired ===")
        
        caplog.clear()
        
        # Check dismissed == false - should find alert1 only
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        assert len(alerts_dto) == 1
        assert alerts_dto[0].fingerprint == alert1.fingerprint
        print(f"✓ Alert1 correctly expired and appears in non-dismissed filter")
        
        # Check dismissed == true - should find alert2 and alert3
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == true", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        assert len(alerts_dto) == 2
        dismissed_fingerprints = {alert.fingerprint for alert in alerts_dto}
        assert alert2.fingerprint in dismissed_fingerprints
        assert alert3.fingerprint in dismissed_fingerprints
        print(f"✓ Alert2 and Alert3 still correctly dismissed")
        
        # Travel to 14:45 - alert2 should also have expired, alert3 still dismissed
        frozen_time.tick(timedelta(minutes=30))
        print(f"\n=== Time: {frozen_time.time_to_freeze} - Alert2 should now also have expired ===")
        
        caplog.clear()
        
        # Check dismissed == false - should find alert1 and alert2
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        assert len(alerts_dto) == 2
        not_dismissed_fingerprints = {alert.fingerprint for alert in alerts_dto}
        assert alert1.fingerprint in not_dismissed_fingerprints
        assert alert2.fingerprint in not_dismissed_fingerprints
        print(f"✓ Alert1 and Alert2 both correctly expired and appear in non-dismissed filter")
        
        # Check dismissed == true - should find only alert3 (forever dismissal)
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == true", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        assert len(alerts_dto) == 1
        assert alerts_dto[0].fingerprint == alert3.fingerprint
        print(f"✓ Alert3 still correctly dismissed forever")
        
        # Verify cleanup logs
        assert "Starting cleanup of expired dismissals" in caplog.text
        assert "Successfully updated expired dismissal" in caplog.text
        print(f"✓ Cleanup logs confirm expired dismissals were updated")
        
        print(f"\n🎉 Mixed expiration times test completed successfully!")


@pytest.mark.parametrize("test_app", ["NO_AUTH"], indirect=True)
def test_api_endpoint_time_travel_scenario(
    db_session, client, test_app, create_alert, caplog
):
    """Test API endpoints with actual time travel using freezegun."""
    
    start_time = datetime.datetime(2025, 6, 17, 16, 0, 0, tzinfo=timezone.utc)
    
    with freeze_time(start_time) as frozen_time:
        # Create an alert at 16:00
        alert = create_alert(
            "api-time-travel-alert",
            AlertStatus.FIRING,
            start_time,
            {
                "name": "API Time Travel Alert",
                "severity": "high",
            },
        )
        
        # Dismiss until 16:20 (20 minutes later) via API
        dismiss_until_time = start_time + timedelta(minutes=20)
        dismiss_until_str = dismiss_until_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        response = client.post(
            "/alerts/batch_enrich",
            headers={"x-api-key": "some-key"},
            json={
                "fingerprints": [alert.fingerprint],
                "enrichments": {
                    "dismissed": "true",
                    "dismissedUntil": dismiss_until_str,
                    "note": "API dismissal test"
                },
            },
        )
        assert response.status_code == 200
        
        time.sleep(1)  # Allow processing
        
        print(f"\n=== Time: {frozen_time.time_to_freeze} (Alert dismissed via API until {dismiss_until_time}) ===")
        
        # At 16:00 - alert should be dismissed
        response = client.post(
            "/alerts/query",
            headers={"x-api-key": "some-key"},
            json={
                "cel": "dismissed == true",
                "limit": 100
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["count"] == 1
        assert result["results"][0]["fingerprint"] == alert.fingerprint
        print(f"✓ API confirms alert is dismissed at 16:00")
        
        # Travel to 16:30 - PAST the dismissal time
        frozen_time.tick(timedelta(minutes=30))
        print(f"\n=== Time: {frozen_time.time_to_freeze} (PAST dismissal expiration via API) ===")
        
        caplog.clear()
        
        # Query for non-dismissed alerts - should find our alert
        response = client.post(
            "/alerts/query",
            headers={"x-api-key": "some-key"},
            json={
                "cel": "dismissed == false",
                "limit": 100
            },
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Key test: expired dismissal should appear in non-dismissed results
        assert result["count"] == 1
        found_alert = result["results"][0]
        assert found_alert["fingerprint"] == alert.fingerprint
        assert found_alert["dismissed"] is False
        print(f"✅ API correctly returns expired alert in dismissed == false filter!")
        
        # Verify cleanup happened
        assert "Starting cleanup of expired dismissals" in caplog.text
        print(f"✓ API endpoint triggered cleanup as expected")
        
        print(f"\n🎉 API time travel test completed successfully!")


@pytest.mark.parametrize("test_app", ["NO_AUTH"], indirect=True)
def test_cleanup_function_direct_time_scenarios(
    db_session, test_app, create_alert, caplog
):
    """Test the cleanup function directly with various time scenarios."""
    
    base_time = datetime.datetime(2025, 6, 17, 12, 0, 0, tzinfo=timezone.utc)
    
    with freeze_time(base_time) as frozen_time:
        # Create alerts
        alert1 = create_alert("cleanup-test-1", AlertStatus.FIRING, base_time, {"name": "Cleanup Test 1"})
        alert2 = create_alert("cleanup-test-2", AlertStatus.FIRING, base_time, {"name": "Cleanup Test 2"})
        alert3 = create_alert("cleanup-test-3", AlertStatus.FIRING, base_time, {"name": "Cleanup Test 3"})
        
        enrichment_bl = EnrichmentsBl("keep", db=db_session)
        
        # Set up dismissals with different scenarios
        # Alert1: Expired 1 hour ago
        past_time = base_time - timedelta(hours=1)
        enrichment_bl.enrich_entity(
            fingerprint=alert1.fingerprint,
            enrichments={
                "dismissed": True,
                "dismissedUntil": past_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "note": "Already expired"
            },
            action_type=ActionType.GENERIC_ENRICH,
            action_callee="test_user",
            action_description="Pre-expired dismissal"
        )
        
        # Alert2: Expires in 1 hour
        future_time = base_time + timedelta(hours=1)
        enrichment_bl.enrich_entity(
            fingerprint=alert2.fingerprint,
            enrichments={
                "dismissed": True,
                "dismissedUntil": future_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "note": "Future expiration"
            },
            action_type=ActionType.GENERIC_ENRICH,
            action_callee="test_user",
            action_description="Future dismissal"
        )
        
        # Alert3: Forever dismissal
        enrichment_bl.enrich_entity(
            fingerprint=alert3.fingerprint,
            enrichments={
                "dismissed": True,
                "dismissedUntil": "forever",
                "note": "Never expires"
            },
            action_type=ActionType.GENERIC_ENRICH,
            action_callee="test_user",
            action_description="Forever dismissal"
        )
        
        print(f"\n=== Testing cleanup function directly at {frozen_time.time_to_freeze} ===")
        
        caplog.clear()
        
        # Run cleanup - should only update alert1 (already expired)
        cleanup_expired_dismissals("keep", db_session)
        
        # Verify logs
        assert "Starting cleanup of expired dismissals" in caplog.text
        assert "Found 3 potentially expired dismissals to check" in caplog.text
        assert "Updating expired dismissal for alert" in caplog.text
        assert "Successfully updated expired dismissal" in caplog.text
        assert "Cleanup completed successfully" in caplog.text
        print(f"✓ Cleanup function processed all dismissals correctly")
        
        # Test the state after cleanup
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        # Should find alert1 (was already expired)
        assert len(alerts_dto) == 1
        assert alerts_dto[0].fingerprint == alert1.fingerprint
        print(f"✓ Alert1 correctly cleaned up (was already expired)")
        
        # Move forward 2 hours - now alert2 should also expire
        frozen_time.tick(timedelta(hours=2))
        print(f"\n=== After moving 2 hours forward to {frozen_time.time_to_freeze} ===")
        
        caplog.clear()
        
        # Run cleanup again
        cleanup_expired_dismissals("keep", db_session)
        
        # Now should clean up alert2 as well
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        # Should find alert1 and alert2 (both expired)
        assert len(alerts_dto) == 2
        not_dismissed_fingerprints = {alert.fingerprint for alert in alerts_dto}
        assert alert1.fingerprint in not_dismissed_fingerprints
        assert alert2.fingerprint in not_dismissed_fingerprints
        print(f"✓ Alert2 also correctly cleaned up after time passed")
        
        # Alert3 should still be dismissed (forever)
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == true", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        assert len(alerts_dto) == 1
        assert alerts_dto[0].fingerprint == alert3.fingerprint
        print(f"✓ Alert3 still correctly dismissed forever")
        
        print(f"\n🎉 Direct cleanup function test completed successfully!")


@pytest.mark.parametrize("test_app", ["NO_AUTH"], indirect=True)
def test_edge_cases_with_time_travel(
    db_session, test_app, create_alert, caplog
):
    """Test edge cases using time travel."""
    
    base_time = datetime.datetime(2025, 6, 17, 9, 0, 0, tzinfo=timezone.utc)
    
    with freeze_time(base_time) as frozen_time:
        # Create alerts for edge case testing
        alert_invalid_time = create_alert("invalid-time", AlertStatus.FIRING, base_time, {"name": "Invalid Time"})
        alert_exact_boundary = create_alert("exact-boundary", AlertStatus.FIRING, base_time, {"name": "Exact Boundary"})
        alert_microseconds = create_alert("microseconds", AlertStatus.FIRING, base_time, {"name": "Microseconds Test"})
        
        enrichment_bl = EnrichmentsBl("keep", db=db_session)
        
        # Edge case 1: Invalid dismissedUntil format
        enrichment_bl.enrich_entity(
            fingerprint=alert_invalid_time.fingerprint,
            enrichments={
                "dismissed": True,
                "dismissedUntil": "invalid-date-format",
                "note": "Invalid time format"
            },
            action_type=ActionType.GENERIC_ENRICH,
            action_callee="test_user",
            action_description="Invalid format test"
        )
        
        # Edge case 2: Exact boundary case - expires at exactly current time
        exact_time = base_time + timedelta(minutes=10)
        enrichment_bl.enrich_entity(
            fingerprint=alert_exact_boundary.fingerprint,
            enrichments={
                "dismissed": True,
                "dismissedUntil": exact_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "note": "Exact boundary test"
            },
            action_type=ActionType.GENERIC_ENRICH,
            action_callee="test_user",
            action_description="Boundary test"
        )
        
        # Edge case 3: Test with microseconds precision
        micro_time = base_time + timedelta(minutes=5, microseconds=123456)
        enrichment_bl.enrich_entity(
            fingerprint=alert_microseconds.fingerprint,
            enrichments={
                "dismissed": True,
                "dismissedUntil": micro_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "note": "Microseconds test"
            },
            action_type=ActionType.GENERIC_ENRICH,
            action_callee="test_user",
            action_description="Microseconds test"
        )
        
        print(f"\n=== Testing edge cases at {frozen_time.time_to_freeze} ===")
        
        caplog.clear()
        
        # Run cleanup - should handle invalid format gracefully
        cleanup_expired_dismissals("keep", db_session)
        
        # Should log warning about invalid format
        assert "Failed to parse dismissedUntil" in caplog.text
        print(f"✓ Invalid date format handled gracefully with warning")
        
        # Move to exactly the boundary time for alert_exact_boundary
        frozen_time.tick(timedelta(minutes=10))
        print(f"\n=== At exact boundary time {frozen_time.time_to_freeze} ===")
        
        caplog.clear()
        
        # Run cleanup - should clean up the exact boundary alert
        cleanup_expired_dismissals("keep", db_session)
        
        # Check that exact boundary alert was cleaned up
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        boundary_alert_found = any(alert.fingerprint == alert_exact_boundary.fingerprint for alert in alerts_dto)
        assert boundary_alert_found
        print(f"✓ Exact boundary case handled correctly (>= comparison)")
        
        # Move past microseconds alert expiration
        frozen_time.tick(timedelta(minutes=-5, microseconds=200000))  # Go to 5 min 200ms
        print(f"\n=== Past microseconds boundary {frozen_time.time_to_freeze} ===")
        
        caplog.clear()
        
        cleanup_expired_dismissals("keep", db_session)
        
        # Check microseconds alert was cleaned up
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        
        micro_alert_found = any(alert.fingerprint == alert_microseconds.fingerprint for alert in alerts_dto)
        assert micro_alert_found
        print(f"✓ Microseconds precision handled correctly")
        
        print(f"\n🎉 Edge cases test completed successfully!")


@pytest.mark.parametrize("test_app", ["NO_AUTH"], indirect=True)
def test_performance_with_many_alerts_time_travel(
    db_session, test_app, create_alert, caplog
):
    """Test performance with many alerts using time travel."""
    
    base_time = datetime.datetime(2025, 6, 17, 20, 0, 0, tzinfo=timezone.utc)
    
    with freeze_time(base_time) as frozen_time:
        print(f"\n=== Creating 20 alerts for performance test ===")
        
        alerts = []
        enrichment_bl = EnrichmentsBl("keep", db=db_session)
        
        # Create 20 alerts with various dismissal times
        for i in range(20):
            alert = create_alert(
                f"perf-alert-{i}",
                AlertStatus.FIRING,
                base_time,
                {"name": f"Performance Test Alert {i}", "severity": "warning"}
            )
            alerts.append(alert)
            
            # Mix of dismissal scenarios
            if i < 5:
                # First 5: Expire in 10 minutes
                expire_time = base_time + timedelta(minutes=10)
            elif i < 10:
                # Next 5: Expire in 30 minutes  
                expire_time = base_time + timedelta(minutes=30)
            elif i < 15:
                # Next 5: Already expired (1 hour ago)
                expire_time = base_time - timedelta(hours=1)
            else:
                # Last 5: Forever dismissal
                expire_time = None
                
            if expire_time:
                dismiss_until_str = expire_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                dismiss_until_str = "forever"
                
            enrichment_bl.enrich_entity(
                fingerprint=alert.fingerprint,
                enrichments={
                    "dismissed": True,
                    "dismissedUntil": dismiss_until_str,
                    "note": f"Performance test dismissal {i}"
                },
                action_type=ActionType.GENERIC_ENRICH,
                action_callee="test_user", 
                action_description=f"Performance test {i}"
            )
        
        print(f"✓ Created 20 alerts with mixed dismissal scenarios")
        
        # Test initial state - should have 5 already expired alerts
        caplog.clear()
        
        start_query_time = time.time()
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        query_duration = time.time() - start_query_time
        
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        assert len(alerts_dto) == 5  # The 5 already expired alerts
        
        print(f"✓ Initial query found 5 expired alerts in {query_duration:.3f}s")
        assert "Found 20 potentially expired dismissals to check" in caplog.text
        assert "Cleanup completed successfully" in caplog.text
        
        # Move forward 15 minutes - should expire first batch (5 more)
        frozen_time.tick(timedelta(minutes=15))
        print(f"\n=== After 15 minutes: {frozen_time.time_to_freeze} ===")
        
        caplog.clear()
        
        start_query_time = time.time()
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        query_duration = time.time() - start_query_time
        
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        assert len(alerts_dto) == 10  # 5 originally expired + 5 newly expired
        
        print(f"✓ After 15min: found 10 expired alerts in {query_duration:.3f}s")
        
        # Move forward another 20 minutes - should expire second batch (5 more)  
        frozen_time.tick(timedelta(minutes=20))
        print(f"\n=== After 35 minutes total: {frozen_time.time_to_freeze} ===")
        
        caplog.clear()
        
        start_query_time = time.time()
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == false", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        query_duration = time.time() - start_query_time
        
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        assert len(alerts_dto) == 15  # All non-forever dismissals expired
        
        print(f"✓ After 35min: found 15 expired alerts in {query_duration:.3f}s")
        
        # Check that 5 alerts are still dismissed (forever dismissals)
        db_alerts, _ = query_last_alerts(
            tenant_id="keep",
            query=QueryDto(cel="dismissed == true", limit=100, sort_by="timestamp", sort_dir="desc", sort_options=[])
        )
        alerts_dto = convert_db_alerts_to_dto_alerts(db_alerts)
        assert len(alerts_dto) == 5  # The forever dismissed alerts
        
        print(f"✓ 5 alerts still correctly dismissed forever")
        print(f"\n🎉 Performance test with 20 alerts completed successfully!")


if __name__ == "__main__":
    # Run the tests individually for debugging
    pass