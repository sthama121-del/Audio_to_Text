import json
import azure.functions as func
from src.app import build_app

app = func.FunctionApp()

GRAPH_APP = build_app()  # cold start init

@app.route(route="copilot_api", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def copilot_api(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        user_query = body.get("query")

        if not user_query:
            return func.HttpResponse(
                "Missing 'query' in JSON body",
                status_code=400
            )

        result = GRAPH_APP.invoke({"user_query": user_query})
        answer = result.get("final_answer") or result.get("draft_answer") or ""

        return func.HttpResponse(
            json.dumps({"answer": answer}, ensure_ascii=False),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        return func.HttpResponse(str(e), status_code=500)
