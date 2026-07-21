import unittest

from evidroid.baselines import (
    abstract_api,
    build_drebin_features,
    build_droidapiminer_features,
    build_mamadroid_features_from_evidence,
    markov_transition_features,
)


class BaselineFeatureTests(unittest.TestCase):
    def test_abstract_api_package(self):
        api = "Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;"
        self.assertEqual(abstract_api(api, abstraction="family"), "android")
        self.assertEqual(abstract_api(api, abstraction="package"), "android.telephony")

    def test_markov_transition_features(self):
        features = markov_transition_features(["android", "java", "android", "java"])
        self.assertEqual(features["mamadroid::android->java"], 1.0)
        self.assertEqual(features["mamadroid::java->android"], 1.0)

    def test_mamadroid_features_from_evidence_family(self):
        evidence = {
            "evidence": [
                {"view": "api", "value": "Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;"},
                {"view": "api", "value": "Ljava/lang/String;->valueOf()Ljava/lang/String;"},
                {"view": "api", "value": "Landroid/location/LocationManager;->getLastKnownLocation()V"},
            ]
        }
        features = build_mamadroid_features_from_evidence(evidence, abstraction="family")
        self.assertEqual(features["mamadroid::android->java"], 1.0)
        self.assertEqual(features["mamadroid::java->android"], 1.0)

    def test_drebin_original_features_filter_apis_and_networks(self):
        evidence = {
            "evidence": [
                {"view": "permission", "value": "android.permission.SEND_SMS"},
                {"view": "api", "value": "Ljava/lang/String;->valueOf()Ljava/lang/String;"},
                {"view": "api", "value": "Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;"},
                {"view": "api", "value": "Ljava/lang/Runtime;->exec(Ljava/lang/String;)Ljava/lang/Process;"},
                {"view": "string", "value": "https://example.com/payload.apk"},
                {"view": "string", "value": "plain sms text"},
            ]
        }
        features = build_drebin_features(evidence)
        self.assertIn("drebin::requested_permission::SEND_SMS", features)
        self.assertFalse(any("String;->valueOf" in key for key in features))
        self.assertTrue(any(key.startswith("drebin::restricted_api::") for key in features))
        self.assertTrue(any(key.startswith("drebin::suspicious_api::") for key in features))
        self.assertTrue(any(key.startswith("drebin::network_address::") for key in features))
        self.assertFalse(any(key.startswith("drebin::suspicious_string::") for key in features))

    def test_droidapiminer_features_use_api_and_permissions_only(self):
        evidence = {
            "evidence": [
                {"view": "permission", "value": "android.permission.INTERNET"},
                {"view": "api", "value": "Landroid/net/Uri;->parse(Ljava/lang/String;)Landroid/net/Uri;"},
                {"view": "component", "value": "activity:com.example.MainActivity"},
                {"view": "string", "value": "https://example.com"},
            ]
        }
        features = build_droidapiminer_features(evidence)
        self.assertIn("droidapiminer::permission::INTERNET", features)
        self.assertTrue(any(key.startswith("droidapiminer::api::") for key in features))
        self.assertFalse(any("component" in key for key in features))
        self.assertFalse(any("example.com" in key for key in features))


if __name__ == "__main__":
    unittest.main()
