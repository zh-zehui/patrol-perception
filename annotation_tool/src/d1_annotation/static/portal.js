const $ = (id) => document.getElementById(id);
const state = { settings: null, classes: [], editingId: null, defaultPrompt: "" };

function notify(text, error = false) {
  const el = $("msg");
  el.textContent = text || "";
  el.className = error ? "msg error" : "msg";
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function ensureDefaultPrompt() {
  if (state.defaultPrompt) return state.defaultPrompt;
  const data = await api("/api/portal/default-prompt");
  state.defaultPrompt = data.prompt || "";
  return state.defaultPrompt;
}

function selectedProfile() {
  const id = $("taskProfile").value;
  return (state.settings?.profiles || []).find((p) => p.id === id) || null;
}

function isLlmProfile(profile) {
  if (!profile) return false;
  return String(profile.type || "openai_compatible") !== "sam3";
}

async function fillDefaultPromptIfNeeded({ force = false } = {}) {
  const profile = selectedProfile();
  if (!force && !isLlmProfile(profile)) return;
  const prompt = await ensureDefaultPrompt();
  if (!prompt) return;
  if (force || !$("taskPrompt").value.trim()) {
    $("taskPrompt").value = prompt;
  }
}

function fillProfileForm(p) {
  $("profId").value = p.id || "";
  $("profName").value = p.name || "";
  $("profType").value = p.type || "openai_compatible";
  $("profDefault").checked = p.id === state.settings?.default_profile_id;
  toggleProfileFields();
  if ((p.type || "") === "sam3") {
    $("profSamUrl").value = p.service_url || "http://127.0.0.1:8783";
    $("profSamModel").value = p.model || "sam3";
    $("profSamConf").value = p.confidence_threshold ?? 0.5;
  } else {
    $("profBaseUrl").value = p.base_url || "https://api.example.com/v1";
    $("profWireApi").value = p.wire_api || "responses";
    $("profModel").value = p.model || "";
    $("profKeyId").value = p.key_id || "";
    $("profEffort").value = p.reasoning_effort || "medium";
  }
  $("profId").focus();
  notify(`已载入档案 ${p.id}，修改后点击「保存模型档案」`);
}


async function api(url, opt = {}) {
  const r = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(opt.headers || {}) },
    ...opt,
  });
  let data = {};
  try { data = await r.json(); } catch {}
  if (!r.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((x) => x.msg || JSON.stringify(x)).join("; ")
      : (data.detail || r.statusText);
    throw Error(detail);
  }
  return data;
}

function fillSelect(sel, items, placeholder, valueKey = "id", labelFn = null) {
  const current = sel.value;
  sel.innerHTML = `<option value="">${placeholder}</option>` + items.map((item) => {
    const value = item[valueKey];
    const label = labelFn ? labelFn(item) : `${item.name || value} (${value})`;
    return `<option value="${value}">${label}</option>`;
  }).join("");
  if ([...sel.options].some((o) => o.value === current)) sel.value = current;
}

function renderKeys() {
  const keys = state.settings?.keys || [];
  $("keyRows").innerHTML = keys.map((k) => `<tr>
    <td><code>${k.id}</code></td><td>${k.name || ""}</td>
    <td>${k.api_key_masked || ""}</td><td>${k.enabled ? "启用" : "停用"}</td>
    <td><button type="button" data-del-key="${k.id}">删除</button></td>
  </tr>`).join("") || `<tr><td colspan="5" class="muted">暂无 Key</td></tr>`;
  fillSelect($("profKeyId"), keys, "选择绑定 Key");
  fillSelect($("taskKey"), keys, "可选：覆盖 Key");
}

function renderProfiles() {
  const profiles = state.settings?.profiles || [];
  const defaultId = state.settings?.default_profile_id || "";
  $("defaultProfileHint").textContent = defaultId ? `当前默认：${defaultId}` : "尚未设置默认档案";
  $("profileRows").innerHTML = profiles.map((p) => {
    const detail = p.type === "sam3"
      ? `${p.service_url || ""} / ${p.model || ""}`
      : `${p.model || ""} @ ${p.base_url || ""} (${p.wire_api || ""})`;
    const star = p.id === defaultId ? " ★" : "";
    const keyHint = p.key_id ? ` · key=${p.key_id}` : (p.type === "sam3" ? "" : " · 未绑 Key");
    return `<tr>
      <td><code>${p.id}</code>${star}</td><td>${p.name || ""}</td><td>${p.type}</td><td>${detail}${keyHint}</td>
      <td><div class="row-actions">
        <button type="button" data-edit-profile="${p.id}">编辑</button>
        <button type="button" data-del-profile="${p.id}">删除</button>
      </div></td>
    </tr>`;
  }).join("") || `<tr><td colspan="5" class="muted">暂无模型档案</td></tr>`;
  fillSelect($("taskProfile"), profiles, "选择预标注模型档案");
  if (defaultId && !$("taskProfile").value) $("taskProfile").value = defaultId;
}

function renderClassEditor() {
  $("classEditor").innerHTML = state.classes.map((c, i) => `<div class="class-row">
    <input data-i="${i}" data-f="id" value="${c.id}" title="id">
    <input data-i="${i}" data-f="name" value="${c.name || ""}" placeholder="name">
    <input data-i="${i}" data-f="color" value="${c.color || "#22C55E"}" placeholder="color">
    <input data-i="${i}" data-f="prompts" value="${(c.prompts || []).join(", ")}" placeholder="prompts，逗号分隔（SAM3/LLM）">
    <button type="button" data-del-class="${i}">删除</button>
  </div>`).join("") || `<div class="muted">还没有类别，请“+ 类别”或载入模板</div>`;
}

function readClassesFromDom() {
  const rows = [...$("classEditor").querySelectorAll(".class-row")];
  return rows.map((row) => {
    const get = (f) => row.querySelector(`[data-f="${f}"]`)?.value?.trim() || "";
    const prompts = get("prompts").split(/[,，]/).map((x) => x.trim()).filter(Boolean);
    return {
      id: Number(get("id") || 0),
      name: get("name"),
      color: get("color") || "#22C55E",
      prompts,
    };
  });
}

async function loadAll() {
  notify("正在加载…");
  state.settings = await api("/api/portal/settings");
  try { await ensureDefaultPrompt(); } catch {}
  renderKeys();
  renderProfiles();
  const ds = await api("/api/portal/datasets");
  $("datasetRows").innerHTML = (ds.datasets || []).map((d) => `<tr>
    <td><strong>${d.name}</strong><br><small>${d.id}</small><div class="muted">${d.notes || ""}</div></td>
    <td><code>${d.path}</code></td>
    <td>${d.model_profile_id || "-"}</td>
    <td>${d.class_count ?? "-"}</td>
    <td><div class="row-actions">
      <button type="button" data-edit="${d.id}">编辑配置</button>
      <button type="button" data-start="${d.id}">开始标注</button>
      <button type="button" data-start-rescan="${d.id}">重扫后标注</button>
      <button type="button" class="danger" data-reset-ann="${d.id}">重置标注</button>
      <button type="button" data-del-ds="${d.id}">删除记录</button>
    </div></td>
  </tr>`).join("") || `<tr><td colspan="5" class="muted">暂无数据集</td></tr>`;
  notify(`已加载（配置目录：${ds.settings_dir || state.settings.settings_dir}）`);
  await fillDefaultPromptIfNeeded();
}

function toggleProfileFields() {
  const sam = $("profType").value === "sam3";
  $("profSamFields").hidden = !sam;
  $("profLlmFields").hidden = sam;
}

function resetTaskForm() {
  state.editingId = null;
  $("taskFormTitle").textContent = "注册数据集任务";
  $("taskId").disabled = false;
  $("taskId").value = "";
  $("taskName").value = "";
  $("taskPath").value = "";
  $("taskNotes").value = "";
  $("taskPrompt").value = "";
  $("taskConf").value = "0.5";
  $("taskKey").value = "";
  if (state.settings?.default_profile_id) $("taskProfile").value = state.settings.default_profile_id;
  state.classes = [];
  renderClassEditor();
  fillDefaultPromptIfNeeded().catch(() => {});
}

async function editTask(id) {
  const data = await api(`/api/portal/datasets/${encodeURIComponent(id)}`);
  state.editingId = id;
  $("taskFormTitle").textContent = `编辑任务：${id}`;
  $("taskId").value = data.task.id || id;
  $("taskId").disabled = true;
  $("taskName").value = data.task.name || data.dataset.name || "";
  $("taskPath").value = data.task.path || data.dataset.path || "";
  $("taskNotes").value = data.task.notes || "";
  $("taskPrompt").value = data.task.system_prompt || "";
  $("taskConf").value = data.task.confidence_threshold ?? 0.5;
  $("taskProfile").value = data.task.model_profile_id || "";
  $("taskKey").value = data.task.key_id || "";
  state.classes = data.classes || [];
  renderClassEditor();
  if (!$("taskPrompt").value.trim()) {
    await fillDefaultPromptIfNeeded();
  }
  $("taskFormCard").scrollIntoView({ behavior: "smooth" });
}

async function saveTask() {
  const payload = {
    id: $("taskId").value.trim(),
    name: $("taskName").value.trim(),
    path: $("taskPath").value.trim(),
    notes: $("taskNotes").value,
    model_profile_id: $("taskProfile").value,
    key_id: $("taskKey").value,
    system_prompt: $("taskPrompt").value,
    confidence_threshold: Number($("taskConf").value || 0.5),
    classes: readClassesFromDom(),
    source_type: "images_dataset",
  };
  if (state.editingId) {
    await api(`/api/portal/datasets/${encodeURIComponent(state.editingId)}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    notify(`已更新任务 ${state.editingId}`);
  } else {
    await api("/api/portal/datasets", { method: "POST", body: JSON.stringify(payload) });
    notify(`已注册任务 ${payload.id}`);
  }
  await loadAll();
}

document.addEventListener("click", async (e) => {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return;
  try {
    if (t.id === "btnReload") return void (await loadAll());
    if (t.id === "btnSaveKey") {
      await api("/api/portal/keys", {
        method: "POST",
        body: JSON.stringify({
          id: $("keyId").value.trim(),
          name: $("keyName").value.trim(),
          api_key: $("keyValue").value,
          note: $("keyNote").value,
          enabled: true,
        }),
      });
      $("keyValue").value = "";
      notify("Key 已保存");
      return void (await loadAll());
    }
    if (t.dataset.delKey) {
      if (!confirm(`删除 Key ${t.dataset.delKey}?`)) return;
      await api(`/api/portal/keys/${encodeURIComponent(t.dataset.delKey)}`, { method: "DELETE" });
      return void (await loadAll());
    }
    if (t.id === "btnSaveProfile") {
      const type = $("profType").value;
      const id = $("profId").value.trim();
      const name = $("profName").value.trim();
      if (!id) {
        notify("请填写档案 ID（例如要更新已有档案，先点表格里的「编辑」，或手动填 sribd-gpt56）", true);
        $("profId").focus();
        return;
      }
      if (!/^[A-Za-z0-9_-]+$/.test(id)) {
        notify("档案 ID 只能包含字母、数字、下划线和连字符", true);
        $("profId").focus();
        return;
      }
      const body = {
        id,
        name: name || id,
        type,
        enabled: true,
        make_default: $("profDefault").checked,
      };
      if (type === "sam3") {
        Object.assign(body, {
          service_url: $("profSamUrl").value.trim(),
          model: $("profSamModel").value.trim() || "sam3",
          confidence_threshold: Number($("profSamConf").value || 0.5),
          request_timeout_sec: 600,
        });
        if (!body.service_url) {
          notify("SAM3 档案需要填写 service_url", true);
          return;
        }
      } else {
        Object.assign(body, {
          base_url: $("profBaseUrl").value.trim(),
          wire_api: $("profWireApi").value,
          model: $("profModel").value.trim(),
          key_id: $("profKeyId").value,
          reasoning_effort: $("profEffort").value,
          store: false,
          request_timeout_sec: 180,
        });
        if (!body.base_url || !body.model) {
          notify("大模型档案需要填写 base_url 和 model", true);
          return;
        }
        if (!body.key_id) {
          notify("请选择绑定 Key（否则只能依赖环境变量 OPENAI_API_KEY）", true);
          // still allow save, but warn — user asked for reaction; better block for clarity
          return;
        }
      }
      notify("正在保存模型档案…");
      await api("/api/portal/profiles", { method: "POST", body: JSON.stringify(body) });
      notify(`模型档案已保存：${id}`);
      return void (await loadAll());
    }
    if (t.dataset.editProfile) {
      const profile = (state.settings?.profiles || []).find((p) => p.id === t.dataset.editProfile);
      if (!profile) {
        notify(`找不到档案 ${t.dataset.editProfile}`, true);
        return;
      }
      fillProfileForm(profile);
      return;
    }
    if (t.dataset.delProfile) {
      if (!confirm(`删除模型档案 ${t.dataset.delProfile}?`)) return;
      await api(`/api/portal/profiles/${encodeURIComponent(t.dataset.delProfile)}`, { method: "DELETE" });
      return void (await loadAll());
    }
    if (t.dataset.edit) return void (await editTask(t.dataset.edit));
    if (t.dataset.start || t.dataset.startRescan) {
      const id = t.dataset.start || t.dataset.startRescan;
      const rescan = Boolean(t.dataset.startRescan);
      notify(`正在启动 ${id}…`);
      const d = await api(`/api/portal/datasets/${encodeURIComponent(id)}/start`, {
        method: "POST",
        body: JSON.stringify({ rescan }),
      });
      location.href = d.url;
      return;
    }
    if (t.dataset.resetAnn) {
      const id = t.dataset.resetAnn;
      if (!confirm(
        `将清空数据集「${id}」的全部标注结果（.d1_annotations/ 与 labels/）。\n` +
        `图片与任务配置会保留，且无法恢复。确认继续？`
      )) return;
      notify(`正在重置 ${id}…`);
      const result = await api(`/api/portal/datasets/${encodeURIComponent(id)}/reset-annotations`, {
        method: "POST",
        body: "{}",
      });
      const parts = (result.deleted || []).length
        ? `已删除：${result.deleted.join(", ")}`
        : "无可删标注产物（可能本就未标注）";
      notify(`${id} 已恢复为未标注状态；${parts}${result.stopped ? "；已停止标注服务" : ""}`);
      return void (await loadAll());
    }
    if (t.dataset.delDs) {
      if (!confirm(`删除数据集记录 ${t.dataset.delDs}?（不删除磁盘数据）`)) return;
      await api(`/api/portal/datasets/${encodeURIComponent(t.dataset.delDs)}`, { method: "DELETE" });
      return void (await loadAll());
    }
    if (t.id === "btnAddClass") {
      state.classes = readClassesFromDom();
      state.classes.push({
        id: state.classes.length,
        name: `class_${state.classes.length}`,
        color: "#3B82F6",
        prompts: [],
      });
      return void renderClassEditor();
    }
    if (t.dataset.delClass !== undefined) {
      state.classes = readClassesFromDom();
      state.classes.splice(Number(t.dataset.delClass), 1);
      state.classes = state.classes.map((c, i) => ({ ...c, id: i }));
      return void renderClassEditor();
    }
    if (t.id === "btnLoadTemplate") {
      const data = await api("/api/portal/class-template");
      state.classes = data.classes || [];
      renderClassEditor();
      await fillDefaultPromptIfNeeded();
      return void notify("已载入默认类别模板（若为空也已尝试载入默认提示词）");
    }
    if (t.id === "btnLoadPrompt") {
      await fillDefaultPromptIfNeeded({ force: true });
      return void notify("已载入 autolabel.py 默认提示词，可直接修改后保存");
    }
    if (t.id === "btnSaveTask") return void (await saveTask());
    if (t.id === "btnResetTask") return void resetTaskForm();
  } catch (err) {
    notify(err.message || String(err), true);
  }
});

$("profType").addEventListener("change", toggleProfileFields);
$("taskProfile").addEventListener("change", () => {
  fillDefaultPromptIfNeeded().catch((e) => notify(e.message || String(e), true));
});
toggleProfileFields();
loadAll().catch((e) => notify(e.message, true));
