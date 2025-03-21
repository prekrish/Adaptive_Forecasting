from typing import Dict, Optional, Tuple
import mlflow

def log_models_to_mlflow(
    model_results: Dict,
    experiment_name: str,
    run_name_prefix: str,
    model_name: str,
    mlflow_uri: str = "http://localhost:5000"
) -> None:
    """
    Log models, parameters, and metrics to MLflow.
    Only registers the model marked as best model.
    
    Parameters
    ----------
    model_results : dict
        Dictionary containing results from all models including individual and ensemble models
    experiment_name : str
        Name of the MLflow experiment
    run_name_prefix : str
        Prefix for run names (will be combined with algorithm name)
    model_name : str
        Name to use when registering the best model
    mlflow_uri : str, default="http://localhost:5000"
        MLflow tracking server URI
    """
    import mlflow
    import json
    from datetime import datetime
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error setting up MLflow experiment: {e}")
        return
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    # Log each model
    for algo_name, results in model_results.items():
        run_name = f"{run_name_prefix}_{algo_name}"
        is_best_model = results.get('is_best_model', 0) == 1
        
        try:
            with mlflow.start_run(run_name=run_name):
                # Log basic information
                mlflow.set_tag("algorithm", algo_name)
                mlflow.set_tag("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                mlflow.set_tag("result_id", results.get('result_id', 'N/A'))
                mlflow.set_tag("is_best_model", is_best_model)
                
                # Log best parameters
                if 'best_params' in results:
                    # Convert any non-serializable types to strings
                    best_params = {}
                    for k, v in results['best_params'].items():
                        if isinstance(v, (int, float, str, bool)):
                            best_params[k] = v
                        else:
                            best_params[k] = str(v)
                    mlflow.log_params(best_params)
                
                # Log metrics
                if 'metrics' in results:
                    for metric_name, value in results['metrics'].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(metric_name, value)
                
                # Log model weights for ensemble models
                if 'model_weights' in results:
                    weights_file = "model_weights.json"
                    with open(weights_file, 'w') as f:
                        json.dump(results['model_weights'], f)
                    mlflow.log_artifact(weights_file)
                
                # Log the forecaster if available
                if 'best_forecaster' in results:
                    try:
                        if is_best_model:
                            # Register the best model
                            mlflow.sklearn.log_model(
                                results['best_forecaster'],
                                "model",
                                registered_model_name=f"{model_name}"
                            )
                        else:
                            # Log without registering
                            mlflow.sklearn.log_model(
                                results['best_forecaster'],
                                "model"
                            )
                    except Exception as e:
                        print(f"Warning: Could not log model for {algo_name}: {e}")
                
                # Log predictions as JSON
                for pred_type in ['test_predictions', 'future_predictions']:
                    if pred_type in results and results[pred_type] is not None:
                        pred_data = results[pred_type]
                        # Convert predictions to dictionary with index
                        pred_dict = {
                            'index': pred_data.index.astype(str).tolist(),
                            'values': pred_data.values.tolist()
                        }
                        pred_file = f"{pred_type}.json"
                        with open(pred_file, 'w') as f:
                            json.dump(pred_dict, f)
                        mlflow.log_artifact(pred_file)
                
                print(f"Successfully logged {algo_name} to MLflow" + 
                      " (Best Model)" if is_best_model else "")
                
        except Exception as e:
            print(f"Error logging {algo_name} to MLflow: {e}")
            continue

def load_model_from_mlflow(
    experiment_name: str,
    run_name: str,
    mlflow_uri: str = "http://localhost:5000"
) -> Tuple[object, Dict]:
    """
    Load a model and its associated artifacts from MLflow.
    
    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment
    run_name : str
        Name of the run to load
    mlflow_uri : str, default="http://localhost:5000"
        MLflow tracking server URI
        
    Returns
    -------
    tuple
        - loaded_model: The loaded forecaster model
        - artifacts: Dictionary containing loaded artifacts
    """
    import mlflow
    import json
    
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")
    
    # Find the run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    
    if len(runs) == 0:
        raise ValueError(f"Run {run_name} not found")
    
    run_id = runs.iloc[0].run_id
    
    # Load the model
    loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    
    # Load artifacts
    artifacts = {}
    client = mlflow.tracking.MlflowClient()
    
    # Load predictions
    for artifact_name in ['test_predictions.json', 'future_predictions.json', 'model_weights.json']:
        try:
            local_path = client.download_artifacts(run_id, artifact_name)
            with open(local_path, 'r') as f:
                artifacts[artifact_name.replace('.json', '')] = json.load(f)
        except:
            continue
    
    return loaded_model, artifacts