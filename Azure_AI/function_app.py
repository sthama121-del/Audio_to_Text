"""
Azure Function: Approval Webhook Handler
This function handles approval/rejection signals from email links

DEPLOYMENT LOCATION: Azure Functions (HTTP Trigger)
RUNTIME: Python 3.10+
"""

import logging
import json
import azure.functions as func
from azure.data.tables import TableServiceClient
from azure.identity import DefaultAzureCredential

app = func.FunctionApp()

# Azure Table Storage for approval state management
TABLE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
TABLE_NAME = "ApprovalRequests"

@app.route(route="approval", auth_level=func.AuthLevel.ANONYMOUS)
def approval_handler(req: func.HttpRequest) -> func.HttpResponse:
    """
    Handle approval/rejection webhook from email
    
    Query Parameters:
        - id: approval_id
        - action: 'approve' or 'reject'
    """
    logging.info('Approval webhook triggered')
    
    try:
        # Get parameters
        approval_id = req.params.get('id')
        action = req.params.get('action')
        
        if not approval_id or not action:
            return func.HttpResponse(
                "Missing required parameters: id and action",
                status_code=400
            )
        
        if action not in ['approve', 'reject']:
            return func.HttpResponse(
                "Invalid action. Must be 'approve' or 'reject'",
                status_code=400
            )
        
        # Update approval status in Azure Table Storage
        table_service = TableServiceClient.from_connection_string(
            TABLE_CONNECTION_STRING
        )
        table_client = table_service.get_table_client(TABLE_NAME)
        
        # Retrieve existing approval request
        try:
            entity = table_client.get_entity(
                partition_key="approval",
                row_key=approval_id
            )
        except Exception as e:
            logging.error(f"Approval ID not found: {approval_id}")
            return func.HttpResponse(
                "Approval ID not found or already processed",
                status_code=404
            )
        
        # Update status
        entity['status'] = 'approved' if action == 'approve' else 'rejected'
        entity['decision_timestamp'] = datetime.utcnow().isoformat()
        
        table_client.update_entity(entity, mode='replace')
        
        # Return success page
        html_response = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Action Recorded</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                    text-align: center;
                    max-width: 500px;
                }}
                .success-icon {{
                    font-size: 72px;
                    margin-bottom: 20px;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 10px;
                }}
                p {{
                    color: #666;
                    line-height: 1.6;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-icon">{'✓' if action == 'approve' else '✗'}</div>
                <h1>Action Recorded Successfully</h1>
                <p>
                    You have <strong>{'approved' if action == 'approve' else 'rejected'}</strong> 
                    the Databricks job recovery action.
                </p>
                <p>
                    <strong>Approval ID:</strong> {approval_id}
                </p>
                <p style="margin-top: 30px; font-size: 14px; color: #999;">
                    The automated workflow will now proceed with your decision.
                </p>
            </div>
        </body>
        </html>
        """
        
        return func.HttpResponse(
            html_response,
            status_code=200,
            mimetype="text/html"
        )
        
    except Exception as e:
        logging.error(f"Error processing approval: {str(e)}")
        return func.HttpResponse(
            f"Error processing approval: {str(e)}",
            status_code=500
        )


@app.route(route="check-approval", auth_level=func.AuthLevel.FUNCTION)
def check_approval_status(req: func.HttpRequest) -> func.HttpResponse:
    """
    Check approval status (called by the main workflow)
    
    Query Parameters:
        - id: approval_id
    """
    logging.info('Check approval status triggered')
    
    try:
        approval_id = req.params.get('id')
        
        if not approval_id:
            return func.HttpResponse(
                json.dumps({"error": "Missing approval_id"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Retrieve from Azure Table Storage
        table_service = TableServiceClient.from_connection_string(
            TABLE_CONNECTION_STRING
        )
        table_client = table_service.get_table_client(TABLE_NAME)
        
        try:
            entity = table_client.get_entity(
                partition_key="approval",
                row_key=approval_id
            )
            
            response = {
                "approval_id": approval_id,
                "status": entity.get('status', 'pending'),
                "job_id": entity.get('job_id'),
                "run_id": entity.get('run_id'),
                "created_at": entity.get('timestamp'),
                "decision_timestamp": entity.get('decision_timestamp')
            }
            
            return func.HttpResponse(
                json.dumps(response),
                status_code=200,
                mimetype="application/json"
            )
            
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"error": "Approval not found"}),
                status_code=404,
                mimetype="application/json"
            )
        
    except Exception as e:
        logging.error(f"Error checking approval: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
