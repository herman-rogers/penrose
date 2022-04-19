defmodule PenroseWeb.PageController do
  use PenroseWeb, :controller

  def index(conn, _params) do
    render(conn, "index.html")
  end
end
