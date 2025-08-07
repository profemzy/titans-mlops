#!/usr/bin/env python3
"""
Quick test script for Titans Finance ML Engineering API

This script performs basic functionality tests to ensure all components
are working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

async def test_model_service():
    """Test model service initialization"""
    try:
        from ai_engineering.api.services import get_model_service

        print("🧪 Testing Model Service...")
        model_service = await get_model_service()

        print(f"✅ Model service loaded: {model_service.is_loaded}")
        print(f"✅ Models available: {len(model_service.models)}")
        print(f"✅ Model config: {list(model_service.model_config.keys())}")

        # Test basic prediction
        test_transaction = {
            "date": "2024-01-15",
            "type": "Expense",
            "description": "Coffee shop",
            "amount": -4.50,
            "payment_method": "credit_card"
        }

        # Test category prediction
        result = await model_service.predict_category(test_transaction)
        print(f"✅ Category prediction: {result['predicted_category']} (confidence: {result['confidence_score']:.2f})")

        # Test amount prediction
        result = await model_service.predict_amount(test_transaction)
        print(f"✅ Amount prediction: ${result['predicted_amount']:.2f}")

        # Test anomaly detection
        result = await model_service.detect_anomaly(test_transaction)
        print(f"✅ Anomaly detection: {'Anomaly' if result['is_anomaly'] else 'Normal'} (score: {result['anomaly_score']:.3f})")

        return True

    except Exception as e:
        print(f"❌ Model service test failed: {e}")
        return False

async def test_feature_processor():
    """Test feature processor"""
    try:
        from ai_engineering.api.services import get_feature_processor

        print("\n🧪 Testing Feature Processor...")
        feature_processor = await get_feature_processor()

        print(f"✅ Feature processor initialized: {feature_processor.is_initialized}")

        # Test feature processing
        test_transaction = {
            "date": "2024-01-15T10:30:00",
            "type": "Expense",
            "description": "Coffee shop purchase",
            "amount": -4.50,
            "payment_method": "credit_card",
            "category": "Food & Dining"
        }

        # Test validation
        validation = await feature_processor.validate_input_data(test_transaction)
        print(f"✅ Data validation passed: {validation['is_valid']}")
        if validation['warnings']:
            print(f"⚠️ Validation warnings: {validation['warnings']}")

        # Test feature processing
        features = await feature_processor.process_transaction_features(test_transaction)
        print(f"✅ Features generated: {len(features)} features")
        print(f"✅ Feature names available: {len(feature_processor.get_feature_names())} names")

        return True

    except Exception as e:
        print(f"❌ Feature processor test failed: {e}")
        return False

async def test_api_schemas():
    """Test API schemas"""
    try:
        from ai_engineering.api.models import (
            TransactionInput,
            EnhancedCategoryResponse,
            EnhancedAmountResponse,
            EnhancedAnomalyResponse
        )

        print("\n🧪 Testing API Schemas...")

        # Test transaction input schema
        test_data = {
            "date": "2024-01-15T10:30:00",
            "type": "Expense",
            "description": "Coffee shop purchase",
            "amount": -4.50,
            "payment_method": "credit_card"
        }

        transaction = TransactionInput(**test_data)
        print(f"✅ TransactionInput schema validation passed")
        print(f"✅ Transaction date: {transaction.date}")
        print(f"✅ Transaction amount: {transaction.amount}")

        # Test response schemas
        category_response = EnhancedCategoryResponse(
            predicted_category="Food & Dining",
            confidence_score=0.95,
            model_version="1.0.0",
            processing_time_ms=25.5,
            features_used=45
        )
        print(f"✅ CategoryResponse schema validation passed")

        return True

    except Exception as e:
        print(f"❌ API schemas test failed: {e}")
        return False

async def test_routes():
    """Test route imports"""
    try:
        from ai_engineering.api.routes import prediction_router, model_router

        print("\n🧪 Testing Routes...")

        print(f"✅ Prediction router loaded: {len(prediction_router.routes)} routes")
        print(f"✅ Model router loaded: {len(model_router.routes)} routes")

        # Print available routes
        print("Available prediction routes:")
        for route in prediction_router.routes:
            print(f"  - {route.methods} {route.path}")

        print("Available model routes:")
        for route in model_router.routes:
            print(f"  - {route.methods} {route.path}")

        return True

    except Exception as e:
        print(f"❌ Routes test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Titans Finance ML Engineering API Tests\n")

    tests = [
        test_api_schemas,
        test_feature_processor,
        test_model_service,
        test_routes
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    print(f"\n📊 Test Summary:")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! ML Engineering API is ready!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
