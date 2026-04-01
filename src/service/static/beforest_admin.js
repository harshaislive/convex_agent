document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("form[data-live-search='true']");
  const toast = document.getElementById("ops-toast");
  let toastTimer = null;

  function showToast(message, isError) {
    if (!toast) return;
    toast.textContent = message;
    toast.style.background = isError ? "#b42318" : "#191919";
    toast.classList.add("show");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(function () {
      toast.classList.remove("show");
    }, 1800);
  }

  function bindRowForms(scope) {
    scope.querySelectorAll("form.conversation-row").forEach(function (rowForm) {
      if (rowForm.dataset.bound === "true") return;
      rowForm.dataset.bound = "true";
      rowForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        const submitter = event.submitter;
        if (!submitter) return;
        const formData = new FormData(rowForm);
        formData.set("status", submitter.value);
        try {
          const response = await fetch(rowForm.action, {
            method: "POST",
            body: formData,
            headers: {
              "x-requested-with": "fetch",
            },
          });
          const payload = await response.json();
          if (!response.ok || !payload.ok) {
            showToast(payload.error || "Could not update handover status.", true);
            return;
          }
          rowForm.querySelectorAll(".toggle").forEach(function (button) {
            button.classList.toggle("active", button.value === payload.handover_status);
          });
          showToast("Status updated", false);
        } catch (_error) {
          showToast("Could not update handover status.", true);
        }
      });
    });
  }

  async function refreshConversationList(query) {
    const target = document.getElementById("conversation-list");
    if (!target) return;
    const url = new URL(window.location.href);
    url.searchParams.set("q", query);
    url.searchParams.delete("contact_id");
    url.searchParams.delete("message");
    url.searchParams.delete("error");
    try {
      const response = await fetch(url.toString(), {
        headers: {
          "x-requested-with": "fetch",
        },
      });
      if (!response.ok) {
        showToast("Could not refresh inbox.", true);
        return;
      }
      const html = await response.text();
      const doc = new DOMParser().parseFromString(html, "text/html");
      const nextList = doc.getElementById("conversation-list");
      if (!nextList) {
        showToast("Could not refresh inbox.", true);
        return;
      }
      target.innerHTML = nextList.innerHTML;
      window.history.replaceState({}, "", url.toString());
      bindRowForms(target);
    } catch (_error) {
      showToast("Could not refresh inbox.", true);
    }
  }

  if (form) {
    const input = form.querySelector("input[name='q']");
    let timer = null;
    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      const currentInput = form.querySelector("input[name='q']");
      await refreshConversationList(currentInput ? currentInput.value : "");
    });
    if (input) {
      input.addEventListener("input", function () {
        window.clearTimeout(timer);
        timer = window.setTimeout(function () {
          refreshConversationList(input.value);
        }, 220);
      });
    }
  }

  bindRowForms(document);
});
