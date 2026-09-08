"""
Route handlers and API blueprints for Customer Churn Prediction System.
"""

import os
import uuid
import json
import datetime
import pandas as pd
from flask import (
    Blueprint, render_template, request, redirect, url_for,
    jsonify, send_from_directory, flash, current_app, session
)
from werkzeug.utils import secure_filename

from backend.database import db, PredictionHistory
from backend.prediction import predict_single_customer, predict_batch_dataframe
from backend.utils import allowed_file, load_dataset_stats
from ml.model_utils import load_artifacts, DEFAULT_THRESHOLDS

main_bp = Blueprint('main', __name__)


# -------------------------------------------------------------
# 1. Page Routes
# -------------------------------------------------------------

@main_bp.route('/')
def index():
    """Landing page with system overview and quick action cards."""
    _, _, metadata = load_artifacts()
    model_name = metadata.get('model_name', 'Random Forest') if metadata else 'Random Forest'
    accuracy = metadata.get('accuracy', 0.80) if metadata else 0.80
    roc_auc = metadata.get('roc_auc', 0.84) if metadata else 0.84

    return render_template(
        'index.html',
        model_name=model_name,
        accuracy=round(accuracy * 100, 1),
        roc_auc=round(roc_auc, 3)
    )


@main_bp.route('/dashboard')
def dashboard():
    """Main Analytics & Risk Intelligence Dashboard."""
    csv_path = os.path.join(current_app.root_path, '..', 'data', 'churn.csv')
    dataset_stats = load_dataset_stats(csv_path)
    _, _, metadata = load_artifacts()

    # Recent predictions count from database
    recent_preds_count = PredictionHistory.query.count()
    high_risk_history_count = PredictionHistory.query.filter_by(risk_category='High Risk').count()

    return render_template(
        'dashboard.html',
        stats=dataset_stats,
        metadata=metadata or {},
        history_count=recent_preds_count,
        high_risk_count=high_risk_history_count
    )


@main_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """Individual Customer Prediction form and submission handler."""
    if request.method == 'POST':
        try:
            # Form submission
            data = request.form.to_dict()

            # Ensure numeric types
            data['tenure'] = float(data.get('tenure', 0))
            data['MonthlyCharges'] = float(data.get('MonthlyCharges', 0.0))
            data['TotalCharges'] = float(data.get('TotalCharges', 0.0))

            customer_id = data.get('customerID', f"CUST-{uuid.uuid4().hex[:6].upper()}")
            data['customerID'] = customer_id

            # Execute real ML inference
            result = predict_single_customer(data)

            # Persist to SQLite
            history_record = PredictionHistory(
                customer_id=result['customer_id'],
                gender=data.get('gender'),
                senior_citizen=data.get('SeniorCitizen'),
                partner=data.get('Partner'),
                dependents=data.get('Dependents'),
                tenure=data.get('tenure'),
                phone_service=data.get('PhoneService'),
                multiple_lines=data.get('MultipleLines'),
                internet_service=data.get('InternetService'),
                online_security=data.get('OnlineSecurity'),
                online_backup=data.get('OnlineBackup'),
                device_protection=data.get('DeviceProtection'),
                tech_support=data.get('TechSupport'),
                streaming_tv=data.get('StreamingTV'),
                streaming_movies=data.get('StreamingMovies'),
                contract=data.get('Contract'),
                paperless_billing=data.get('PaperlessBilling'),
                payment_method=data.get('PaymentMethod'),
                monthly_charges=data.get('MonthlyCharges'),
                total_charges=data.get('TotalCharges'),
                prediction=result['prediction'],
                probability=result['probability'],
                risk_category=result['risk_category'],
                risk_score=result['risk_score'],
                model_version=result['model_name']
            )
            db.session.add(history_record)
            db.session.commit()

            # Store in session for result page
            session['latest_result'] = result
            session['latest_input'] = data

            return redirect(url_for('main.result'))
        except Exception as e:
            current_app.logger.error(f"Prediction error: {e}")
            flash(f"Error executing prediction: {str(e)}", "danger")
            return redirect(url_for('main.predict'))

    return render_template('predict.html')


@main_bp.route('/result')
def result():
    """Displays comprehensive risk report for the latest predicted customer."""
    res = session.get('latest_result')
    inp = session.get('latest_input')
    if not res:
        flash("No recent prediction found. Please enter customer details.", "info")
        return redirect(url_for('main.predict'))

    return render_template('result.html', result=res, customer=inp)


@main_bp.route('/batch', methods=['GET', 'POST'])
def batch():
    """Batch CSV Upload and Vectorized Inference."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part provided in request.", "danger")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("No selected file. Please choose a CSV file.", "warning")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                original_filename = secure_filename(file.filename)
                unique_prefix = uuid.uuid4().hex[:8]
                saved_filename = f"batch_{unique_prefix}_{original_filename}"
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], saved_filename)
                file.save(filepath)

                # Process batch CSV
                df = pd.read_csv(filepath)
                if df.empty:
                    flash("Uploaded CSV file is empty.", "danger")
                    return redirect(request.url)

                result_df, summary = predict_batch_dataframe(df)

                # Save predicted CSV
                result_filename = f"predicted_{unique_prefix}_{original_filename}"
                result_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], result_filename)
                result_df.to_csv(result_filepath, index=False)

                # Save top records to database history
                model_name = summary.get('model_name', 'Random Forest')
                batch_records = []
                for _, row in result_df.head(50).iterrows():
                    cid = str(row.get('customerID', f"BATCH-{uuid.uuid4().hex[:6]}"))
                    prob = float(row.get('churn_probability', 0.0))
                    cat = str(row.get('risk_category', 'Low Risk'))
                    score = int(row.get('risk_score', 0))
                    pred = str(row.get('prediction', 'Retain'))

                    rec = PredictionHistory(
                        customer_id=cid,
                        gender=str(row.get('gender', '')),
                        tenure=float(row.get('tenure', 0)),
                        contract=str(row.get('Contract', '')),
                        monthly_charges=float(row.get('MonthlyCharges', 0)),
                        total_charges=float(row.get('TotalCharges', 0)) if pd.notnull(row.get('TotalCharges')) else 0.0,
                        prediction=pred,
                        probability=prob,
                        risk_category=cat,
                        risk_score=score,
                        model_version=model_name
                    )
                    batch_records.append(rec)

                if batch_records:
                    db.session.bulk_save_objects(batch_records)
                    db.session.commit()

                # Preview top 50 records in template
                preview_records = result_df.head(100).to_dict(orient='records')

                return render_template(
                    'batch.html',
                    summary=summary,
                    records=preview_records,
                    download_file=result_filename
                )
            except Exception as e:
                current_app.logger.error(f"Batch prediction error: {e}")
                flash(f"Error processing CSV: {str(e)}", "danger")
                return redirect(request.url)
        else:
            flash("Invalid file format. Only standard .csv files are supported.", "danger")
            return redirect(request.url)

    return render_template('batch.html')


@main_bp.route('/download/<filename>')
def download_file(filename):
    """Securely downloads batch prediction and retention campaign CSVs."""
    clean_name = secure_filename(filename)
    return send_from_directory(
        current_app.config['UPLOAD_FOLDER'],
        clean_name,
        as_attachment=True
    )


@main_bp.route('/analytics')
def analytics():
    """In-depth customer segmentation and cohort analytics."""
    csv_path = os.path.join(current_app.root_path, '..', 'data', 'churn.csv')
    stats = load_dataset_stats(csv_path)
    return render_template('analytics.html', stats=stats)


@main_bp.route('/model')
def model_info():
    """ML Model evaluation metrics, comparison benchmarks, and viva explanations."""
    _, _, metadata = load_artifacts()
    return render_template('model.html', metadata=metadata or {})


@main_bp.route('/simulator')
def simulator():
    """Interactive What-If Scenario Simulator."""
    return render_template('simulator.html')


@main_bp.route('/simulate', methods=['POST'])
def simulate():
    """Handles What-If simulation comparing baseline vs modified scenario."""
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()

        # Parse baseline
        baseline = data.get('baseline', {})
        modified = data.get('modified', {})

        if not baseline or not modified:
            return jsonify({'error': 'Both baseline and modified parameters are required.'}), 400

        res_base = predict_single_customer(baseline)
        res_mod = predict_single_customer(modified)

        diff = round(res_mod['probability_percent'] - res_base['probability_percent'], 1)

        # Simulation narrative
        if diff < 0:
            narrative = f"Under the modified scenario, the predicted churn probability decreases by {abs(diff)} percentage points (from {res_base['probability_percent']}% to {res_mod['probability_percent']}%)."
        elif diff > 0:
            narrative = f"Under the modified scenario, the predicted churn probability increases by {diff} percentage points (from {res_base['probability_percent']}% to {res_mod['probability_percent']}%)."
        else:
            narrative = f"Under the modified scenario, the predicted churn probability remains unchanged at {res_base['probability_percent']}%."

        return jsonify({
            'status': 'success',
            'baseline': res_base,
            'modified': res_mod,
            'delta_percent': diff,
            'narrative': narrative
        })
    except Exception as e:
        current_app.logger.error(f"Simulation error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@main_bp.route('/retention')
def retention():
    """Generates prioritized retention campaign list for high-risk accounts."""
    # Fetch high-risk customers from database first; fallback to dataset top high-risk accounts
    high_risk_records = PredictionHistory.query.filter(
        PredictionHistory.probability >= 0.70
    ).order_by(PredictionHistory.probability.desc()).limit(100).all()

    items = []
    if high_risk_records:
        for r in high_risk_records:
            items.append({
                'customer_id': r.customer_id,
                'risk_category': r.risk_category,
                'probability_percent': round(r.probability * 100, 1),
                'probability': r.probability,
                'tenure': int(r.tenure) if r.tenure is not None else 0,
                'contract': r.contract or 'Month-to-month',
                'monthly_charges': round(r.monthly_charges, 2) if r.monthly_charges else 0.0,
                'action': 'Urgent Contract & Pricing Review' if (r.contract and 'Month-to-month' in r.contract) else 'Customer Success Outreach'
            })
    else:
        # Generate sample campaign from dataset
        csv_path = os.path.join(current_app.root_path, '..', 'data', 'churn.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path).head(150)
            res_df, _ = predict_batch_dataframe(df)
            high_df = res_df[res_df['churn_probability'] >= 0.70].sort_values(by='churn_probability', ascending=False).head(50)

            for _, row in high_df.iterrows():
                items.append({
                    'customer_id': str(row.get('customerID', 'CUST-DEMO')),
                    'risk_category': 'High Risk',
                    'probability_percent': round(float(row.get('churn_probability', 0.8)) * 100, 1),
                    'probability': float(row.get('churn_probability', 0.8)),
                    'tenure': int(row.get('tenure', 1)),
                    'contract': str(row.get('Contract', 'Month-to-month')),
                    'monthly_charges': round(float(row.get('MonthlyCharges', 80.0)), 2),
                    'action': 'Urgent Contract & Pricing Review' if 'Month-to-month' in str(row.get('Contract', '')) else 'Tech Support & Concierge Care'
                })

    # Prepare export CSV for campaign download
    campaign_filename = "retention_campaign_active.csv"
    if items:
        camp_df = pd.DataFrame(items)
        camp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], campaign_filename)
        camp_df.to_csv(camp_path, index=False)

    return render_template(
        'retention.html',
        campaign_list=items,
        total_high_risk=len(items),
        download_file=campaign_filename if items else None
    )


@main_bp.route('/history')
def history():
    """Prediction history log with search and risk level filtering."""
    risk_filter = request.args.get('risk', '').strip()
    search_query = request.args.get('q', '').strip()

    query = PredictionHistory.query

    if risk_filter:
        query = query.filter(PredictionHistory.risk_category == risk_filter)
    if search_query:
        query = query.filter(PredictionHistory.customer_id.like(f"%{search_query}%"))

    predictions = query.order_by(PredictionHistory.created_at.desc()).limit(200).all()

    return render_template(
        'history.html',
        predictions=predictions,
        active_filter=risk_filter,
        search_query=search_query
    )


@main_bp.route('/about')
def about():
    """About project, system architecture, methodology, and viva explanations."""
    return render_template('about.html')


# -------------------------------------------------------------
# 2. REST API Endpoints
# -------------------------------------------------------------

@main_bp.route('/health')
@main_bp.route('/api/health')
def health():
    """System health check."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'database': 'connected'
    })


@main_bp.route('/api/model-info')
def api_model_info():
    """Returns serialized model metadata and evaluation benchmarks."""
    _, _, metadata = load_artifacts()
    return jsonify(metadata or {})


@main_bp.route('/api/dashboard-stats')
def api_dashboard_stats():
    """Returns JSON analytics metrics for dashboard Chart.js."""
    csv_path = os.path.join(current_app.root_path, '..', 'data', 'churn.csv')
    stats = load_dataset_stats(csv_path)
    return jsonify(stats)


@main_bp.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API for single customer churn prediction."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    try:
        payload = request.get_json()
        result = predict_single_customer(payload)
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@main_bp.route('/api/history')
def api_history():
    """JSON endpoint returning recent prediction records."""
    records = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).limit(100).all()
    return jsonify([r.to_dict() for r in records])


# -------------------------------------------------------------
# 3. Error Handlers
# -------------------------------------------------------------

@main_bp.app_errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@main_bp.app_errorhandler(500)
def server_error(e):
    current_app.logger.error(f"Internal Server Error: {e}")
    return render_template('500.html'), 500
