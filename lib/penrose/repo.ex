defmodule Penrose.Repo do
  use Ecto.Repo,
    otp_app: :penrose,
    adapter: Ecto.Adapters.Postgres
end
