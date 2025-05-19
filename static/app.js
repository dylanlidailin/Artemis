const API = "https://artem1s-ed2ab9bd3a4a.herokuapp.com";

async function uploadPDF() {
  const file = document.getElementById("pdfInput").files[0];
  if (!file) return alert("Pick a PDF first.");
  const data = new FormData();
  data.append("file", file);
  const res = await fetch(`${API}/upload_pdf`, { method: "POST", body: data });
  const j = await res.json();
  document.getElementById("uploadStatus").textContent = j.status;
}

async function ask() {
  const q = document.getElementById("question").value.trim();
  if (!q) return;
  const msgs = document.getElementById("messages");
  msgs.innerHTML += `<div class="user">You: ${q}</div>`;
  document.getElementById("question").value = "";
  msgs.scrollTop = msgs.scrollHeight;

  const res = await fetch(`${API}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: q })
  });
  const { answer, error } = await res.json();
  msgs.innerHTML += `<div class="bot">${ error || answer }</div>`;
  msgs.scrollTop = msgs.scrollHeight;
}