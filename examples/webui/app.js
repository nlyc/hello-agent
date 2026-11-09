(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const state = {
    loaded: new Set(),
    health: null,
    settings: null,
  };

  // ---------------------------------------------------------------- 基础工具

  const esc = (value) =>
    String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

  async function api(path, { method = 'GET', body } = {}) {
    const options = { method, headers: { Accept: 'application/json' } };
    if (body !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetch(path, options);
    } catch (error) {
      throw new Error(`无法连接服务：${error.message}`);
    }
    let payload = null;
    const text = await response.text();
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = null;
      }
    }
    if (!response.ok) {
      throw new Error((payload && payload.error) || `请求失败（HTTP ${response.status}）`);
    }
    return payload;
  }

  let toastTimer = 0;
  function toast(message, kind = 'ok') {
    const node = $('#toast');
    node.textContent = message;
    node.className = `toast show ${kind}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      node.className = `toast ${kind}`;
    }, kind === 'err' ? 6000 : 3200);
  }

  async function withBusy(button, label, task) {
    const original = button ? button.textContent : '';
    if (button) {
      button.disabled = true;
      button.textContent = label || '处理中…';
    }
    try {
      return await task();
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = original;
      }
    }
  }

  function setHealth(ok, text) {
    const node = $('#health');
    node.className = `pill ${ok ? 'pill-ok' : 'pill-err'}`;
    node.textContent = text;
  }

  function bytes(size) {
    const n = Number(size) || 0;
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  }

  function reason(error) {
    toast(error && error.message ? error.message : String(error), 'err');
    console.error(error);
  }

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = String(reader.result || '');
        const comma = result.indexOf(',');
        resolve(comma >= 0 ? result.slice(comma + 1) : result);
      };
      reader.onerror = () => reject(new Error('读取文件失败'));
      reader.readAsDataURL(file);
    });
  }

  // ------------------------------------------------------------ Markdown 渲染

  function inline(text) {
    return String(text)
      .replace(/&lt;br&gt;/g, '<br>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  }

  function isSeparator(line) {
    const cells = line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|');
    return cells.length > 0 && cells.every((cell) => /^\s*:?-{2,}:?\s*$/.test(cell));
  }

  function splitRow(line) {
    return line
      .trim()
      .replace(/^\|/, '')
      .replace(/\|$/, '')
      .split('|')
      .map((cell) => cell.trim());
  }

  function markdown(source) {
    const lines = String(source ?? '').replace(/\r\n/g, '\n').split('\n');
    const html = [];
    let index = 0;
    let paragraph = [];
    let list = null;

    const flushParagraph = () => {
      if (paragraph.length) {
        html.push(`<p>${inline(paragraph.join(' '))}</p>`);
        paragraph = [];
      }
    };
    const flushList = () => {
      if (list) {
        html.push(`<ul>${list.join('')}</ul>`);
        list = null;
      }
    };
    const flush = () => {
      flushParagraph();
      flushList();
    };

    while (index < lines.length) {
      const line = lines[index];
      const trimmed = line.trim();

      if (trimmed.startsWith('```')) {
        flush();
        index += 1;
        const code = [];
        while (index < lines.length && !lines[index].trim().startsWith('```')) {
          code.push(lines[index]);
          index += 1;
        }
        index += 1;
        html.push(`<pre><code>${esc(code.join('\n'))}</code></pre>`);
        continue;
      }
      if (!trimmed) {
        flush();
        index += 1;
        continue;
      }
      const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed);
      if (heading) {
        flush();
        const level = Math.min(6, heading[1].length);
        html.push(`<h${level}>${inline(esc(heading[2]))}</h${level}>`);
        index += 1;
        continue;
      }
      if (/^(-{3,}|\*{3,})$/.test(trimmed)) {
        flush();
        html.push('<hr>');
        index += 1;
        continue;
      }
      if (trimmed.startsWith('>')) {
        flush();
        const quote = [];
        while (index < lines.length && lines[index].trim().startsWith('>')) {
          quote.push(esc(lines[index].trim().replace(/^>\s?/, '')));
          index += 1;
        }
        html.push(`<blockquote>${inline(quote.join('<br>'))}</blockquote>`);
        continue;
      }
      if (trimmed.includes('|') && index + 1 < lines.length && isSeparator(lines[index + 1])) {
        flush();
        const header = splitRow(trimmed);
        index += 2;
        const rows = [];
        while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
          rows.push(splitRow(lines[index]));
          index += 1;
        }
        const head = header.map((cell) => `<th>${inline(esc(cell))}</th>`).join('');
        const body = rows
          .map((row) => `<tr>${row.map((cell) => `<td>${inline(esc(cell))}</td>`).join('')}</tr>`)
          .join('');
        html.push(`<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`);
        continue;
      }
      const item = /^[-*+]\s+(.*)$/.exec(trimmed) || /^\d+\.\s+(.*)$/.exec(trimmed);
      if (item) {
        flushParagraph();
        if (!list) list = [];
        list.push(`<li>${inline(esc(item[1]))}</li>`);
        index += 1;
        continue;
      }
      flushList();
      paragraph.push(esc(trimmed));
      index += 1;
    }
    flush();
    return html.join('\n');
  }

  function setMarkdown(selector, source, emptyText = '暂无内容。') {
    const node = $(selector);
    const body = markdown(source);
    node.innerHTML = body || `<p class="empty">${esc(emptyText)}</p>`;
  }

  // ---------------------------------------------------------------- 页签切换

  const loaders = {
    overview: loadStatus,
    knowledge: loadDocuments,
    memory: loadMemory,
    paradigms: null,
    demo: loadRuns,
    eval: loadEvalCases,
    tests: loadTests,
  };

  function switchTab(name) {
    if (!name) return;
    $$('#tabs button').forEach((button) => {
      button.classList.toggle('active', button.dataset.tab === name);
    });
    $$('.panel').forEach((panel) => {
      panel.classList.toggle('active', panel.id === `tab-${name}`);
    });
    const loader = loaders[name];
    if (loader && !state.loaded.has(name)) {
      state.loaded.add(name);
      Promise.resolve()
        .then(loader)
        .catch(reason);
    }
  }

  function markFresh(name) {
    state.loaded.add(name);
  }

  // -------------------------------------------------------------------- 概览

  async function loadStatus() {
    const data = await api('/api/status');
    state.health = data;
    setHealth(true, '已连接');

    const rag = data.rag || {};
    const memory = data.memory || {};
    const longTerm = memory.long_term || {};
    const working = memory.working || {};

    $('#stat-docs').textContent = data.documents ?? 0;
    $('#stat-chunks').textContent = rag.chunks ?? 0;
    $('#stat-memories').textContent = longTerm.memories ?? 0;
    $('#stat-working').textContent = working.items ?? 0;
    $('#stat-dim').textContent = longTerm.dimension || 0;
    $('#stat-bm25').textContent = rag.bm25_chunks ?? 0;

    $('#env-namespace').textContent = data.namespace || '—';
    $('#env-user').textContent = data.user_id || '—';
    $('#env-datadir').textContent = data.data_dir || '—';
    $('#env-fingerprint').textContent = rag.embedding_fingerprint || '—';
    $('#env-chunk').textContent = `${rag.chunk_tokens ?? '—'} tokens / overlap ${rag.chunk_overlap ?? '—'}`;
    $('#stat-dim-sub').textContent = (data.settings || {}).embed_backend || '—';

    if (data.settings) renderSettings(data.settings);

    const pill = $('#env-pill');
    pill.className = 'pill pill-ok';
    pill.textContent = '运行中';

    const presets = data.presets || {};
    const questions = presets.questions || [];
    const memoryQueries = presets.memory_queries || [];
    $('#question-presets').innerHTML = questions
      .map((item) => `<option value="${esc(item)}"></option>`)
      .join('');
    $('#memory-presets').innerHTML = memoryQueries
      .map((item) => `<option value="${esc(item)}"></option>`)
      .join('');
    if (presets.main_question) {
      $('#answer-question').placeholder = `例如：${presets.main_question}`;
    }
  }

  // ------------------------------------------------------------ 检索与在线配置

  function renderSettings(settings) {
    state.settings = settings;
    const embedSelect = $('#settings-embed');
    const vectorSelect = $('#settings-vector');
    if (embedSelect) embedSelect.value = settings.embed_backend || 'hashing';
    if (vectorSelect) vectorSelect.value = settings.vector_store || 'sqlite';

    $('#settings-fingerprint').textContent = settings.embedding_fingerprint || '—';

    const llm = settings.llm || {};
    $('#settings-llm').textContent = llm.configured
      ? `已配置 · ${llm.model || '默认模型'}${llm.base_url ? ` · ${llm.base_url}` : ''}`
      : '未配置（在 .env 填写 LLM_API_KEY / LLM_MODEL）';

    const search = settings.search || {};
    const searchFlags = [
      search.tavily_configured ? 'Tavily 已配置' : '',
      search.serpapi_configured ? 'SerpAPI 已配置' : '',
    ].filter(Boolean);
    $('#settings-search').textContent = searchFlags.length
      ? `${searchFlags.join(' · ')} · 最多 ${search.max_results ?? '—'} 条 · 知识库未命中${search.fallback ? '自动联网' : '不联网'}`
      : '未配置（在 .env 填写 TAVILY_API_KEY 或 SERPAPI_API_KEY）';

    const sqlite = settings.sqlite || {};
    const files = sqlite.files || [];
    $('#settings-sqlite').textContent = files.length
      ? `${Number(sqlite.total_mb || 0).toFixed(2)} MB · ${files
          .map((file) => `${file.name} ${bytes(file.bytes)}`)
          .join(' / ')}`
      : '暂无 SQLite 文件';

    const qdrant = settings.qdrant || {};
    $('#settings-qdrant').textContent = qdrant.configured
      ? `${qdrant.url} · 集合 ${qdrant.collection}`
      : '未配置 QDRANT_URL';

    const seedToggle = $('#seed-toggle');
    if (seedToggle) seedToggle.checked = settings.seed_demo !== false;
    $('#seed-hint').textContent = settings.seed_demo === false
      ? '已关闭：重启后不再写入内置文档与演示记忆'
      : '内置：4 篇知识文档 + 2 条演示记忆';

    const pill = $('#settings-pill');
    const error = $('#settings-error');
    if (settings.store_error) {
      pill.className = 'pill pill-err';
      pill.textContent = '已降级';
      error.textContent = settings.store_error;
      error.classList.remove('hidden');
    } else {
      pill.className = 'pill pill-ok';
      pill.textContent = `${settings.embed_backend || '—'} / ${settings.vector_store || '—'}`;
      error.textContent = '';
      error.classList.add('hidden');
    }
  }

  async function loadSettings() {
    const settings = await api('/api/settings');
    renderSettings(settings);
    return settings;
  }

  function switchSummary(result) {
    const parts = [];
    if (!result || result.changed === false) return '配置未变化';
    if (result.migrated_memories) parts.push(`迁移记忆 ${result.migrated_memories} 条`);
    const failed = result.failed_documents || [];
    if (failed.length) parts.push(`需手动重新入库 ${failed.length} 个文档`);
    return parts.join(' · ') || '索引已重建';
  }

  async function applySettings(event) {
    event.preventDefault();
    const button = event.submitter || $('#settings-form button');
    await withBusy(button, '切换中…（重建索引）', async () => {
      const messages = [];
      if (!state.settings) await loadSettings();

      const embed = $('#settings-embed').value;
      if (embed !== state.settings.embed_backend) {
        const result = await api('/api/settings/embed', {
          method: 'POST',
          body: { backend: embed },
        });
        if (result.changed) messages.push(`Embedding → ${result.to}，${switchSummary(result)}`);
        if (result.settings) renderSettings(result.settings);
      }

      const vector = $('#settings-vector').value;
      if (vector !== state.settings.vector_store) {
        const result = await api('/api/settings/vector', {
          method: 'POST',
          body: { backend: vector },
        });
        if (result.changed) messages.push(`向量库 → ${result.to}，${switchSummary(result)}`);
        if (result.settings) renderSettings(result.settings);
      }

      await loadStatus();
      state.loaded.add('overview');
      state.loaded.delete('knowledge');
      state.loaded.delete('memory');
      await refreshDocuments().catch(reason);
      toast(messages.length ? messages.join('；') : '配置未变化');
    }).catch(reason);
  }

  async function toggleSeed(event) {
    const enabled = event.target.checked;
    event.target.disabled = true;
    try {
      const data = await api('/api/settings/seed', {
        method: 'POST',
        body: { enabled },
      });
      if (data.settings) renderSettings(data.settings);
      state.loaded.delete('knowledge');
      state.loaded.delete('memory');
      await loadStatus();
      await refreshDocuments().catch(() => {});
      $('#seed-hint').textContent = enabled
        ? `已启用 · 当前 ${(data.documents || []).length} 篇文档`
        : `已关闭 · 移除 ${data.removed_documents} 篇内置文档、${data.removed_memories} 条记忆条目`;
      toast(enabled ? '已启用内置示例' : '已移除内置示例，仅保留你的数据');
    } catch (error) {
      event.target.checked = !enabled;
      reason(error);
    } finally {
      event.target.disabled = false;
    }
  }

  // ------------------------------------------------------------------ 知识库

  function hitHtml(hit, rank) {
    const citation = hit.citation || hit.chunk_id || '';
    const heading = hit.heading_path || '';
    const meta = [
      hit.doc_id ? `doc: ${hit.doc_id}` : '',
      heading ? `章节: ${heading}` : '',
      hit.page ? `p.${hit.page}` : '',
      hit.source || '',
    ]
      .filter(Boolean)
      .map((part) => esc(part))
      .join(' · ');
    return `
      <article class="hit">
        <div class="hit-head">
          <span class="rank">#${rank}</span>
          <span class="score">score ${Number(hit.score || 0).toFixed(4)}</span>
          <span class="hit-cite">${esc(citation)}</span>
        </div>
        <div class="hit-text">${esc(hit.text || '')}</div>
        <div class="hit-meta">${meta}</div>
      </article>`;
  }

  function renderDocuments(documents) {
    const rows = $('#doc-rows');
    if (!documents || !documents.length) {
      rows.innerHTML = '<tr><td colspan="6" class="empty">还没有文档，上传或从 URL 入库。</td></tr>';
      return;
    }
    rows.innerHTML = documents
      .map((doc) => {
        const docId = String(doc.doc_id || '');
        return `
          <tr>
            <td><div class="doc-title">${esc(doc.title || docId)}</div>
                <div class="doc-source">${esc(doc.source_uri || '')}</div></td>
            <td><span class="chip">${esc(docId)}</span></td>
            <td>${esc(doc.chunk_count ?? 0)}</td>
            <td class="doc-source">${esc(doc.source || '')}</td>
            <td class="doc-source">${esc(doc.ingested_at || '—')}</td>
            <td><button class="row-action" data-delete-doc="${esc(docId)}">删除</button></td>
          </tr>`;
      })
      .join('');
  }

  async function refreshDocuments() {
    const data = await api('/api/documents');
    renderDocuments(data.documents);
    const stats = data.stats || {};
    $('#stat-docs').textContent = (data.documents || []).length;
    $('#stat-chunks').textContent = stats.chunks ?? 0;
    $('#stat-bm25').textContent = stats.bm25_chunks ?? 0;
  }

  async function loadDocuments() {
    await refreshDocuments();
  }

  async function ingestUpload(event) {
    event.preventDefault();
    const input = $('#upload-file');
    const file = input.files && input.files[0];
    if (!file) {
      toast('请选择文件', 'err');
      return;
    }
    await withBusy(event.submitter || $('#upload-form button'), '入库中…', async () => {
      const content = await fileToBase64(file);
      const result = await api('/api/ingest', {
        method: 'POST',
        body: {
          filename: file.name,
          content_base64: content,
          title: $('#upload-title').value.trim() || null,
        },
      });
      renderDocuments(result.documents);
      toast(`已入库 ${result.result.doc_id}（${result.result.chunk_count} chunks）`);
      input.value = '';
      $('#upload-title').value = '';
      await refreshDocuments();
    }).catch(reason);
  }

  async function ingestUrl(event) {
    event.preventDefault();
    const url = $('#url-input').value.trim();
    if (!url) {
      toast('请填写 URL', 'err');
      return;
    }
    await withBusy(event.submitter || $('#url-form button'), '抓取中…', async () => {
      const result = await api('/api/ingest', {
        method: 'POST',
        body: { url, title: $('#url-title').value.trim() || null },
      });
      renderDocuments(result.documents);
      toast(`已抓取入库：${result.result.doc_id}`);
      $('#url-input').value = '';
      $('#url-title').value = '';
      await refreshDocuments();
    }).catch(reason);
  }

  async function deleteDocument(docId) {
    if (!window.confirm(`确定删除文档 ${docId} 的全部索引？`)) return;
    try {
      const result = await api(`/api/documents/${encodeURIComponent(docId)}`, {
        method: 'DELETE',
      });
      renderDocuments(result.documents);
      toast(`已删除 ${docId}（${result.removed_chunks} chunks）`);
      await refreshDocuments();
    } catch (error) {
      reason(error);
    }
  }

  async function clearDocuments() {
    if (!window.confirm('确定清空全部文档索引与清单？磁盘上的源文件不会被删除。')) return;
    try {
      await api('/api/documents/clear', { method: 'POST', body: {} });
      toast('索引已清空');
      await refreshDocuments();
    } catch (error) {
      reason(error);
    }
  }

  async function runSearch(event) {
    event.preventDefault();
    const query = $('#search-query').value.trim();
    if (!query) {
      toast('请输入检索问题', 'err');
      return;
    }
    await withBusy(event.submitter || $('#search-form button'), '检索中…', async () => {
      const data = await api('/api/search', {
        method: 'POST',
        body: {
          query,
          top_k: Number($('#search-topk').value) || 5,
          mode: $('#search-mode').value,
        },
      });
      const hits = data.hits || [];
      $('#search-count').textContent = `${hits.length} 条结果 · ${data.mode}`;
      $('#search-results').innerHTML = hits.length
        ? hits.map((hit, index) => hitHtml(hit, index + 1)).join('')
        : '<p class="empty">没有召回结果，试试换个说法或调大 Top K。</p>';
      toast(`检索完成：${hits.length} 条`);
    }).catch(reason);
  }

  async function runAnswer(event) {
    event.preventDefault();
    const question = $('#answer-question').value.trim();
    if (!question) {
      toast('请输入问题', 'err');
      return;
    }
    await withBusy(event.submitter || $('#answer-form button'), '生成中…', async () => {
      const mode = $('#answer-llm').value;
      const data = await api('/api/answer', {
        method: 'POST',
        body: {
          question,
          top_k: Number($('#answer-topk').value) || 5,
          use_llm: mode !== 'none',
          llm_mode: mode,
        },
      });
      const citations = (data.citations || [])
        .map((item) => `<span>${esc(item)}</span>`)
        .join('');
      const hits = (data.hits || [])
        .map((hit, index) => hitHtml(hit, index + 1))
        .join('');
      const backendLabel = {
        online: `在线 LLM${data.llm_model ? ` · ${data.llm_model}` : ''}`,
        offline: '离线 Demo LLM',
        retrieval: '仅检索（无 LLM）',
      }[data.llm_backend] || data.llm_backend || '—';
      const error = data.llm_error
        ? `<p class="callout">${esc(data.llm_error)}</p>`
        : '';
      const web = data.web_search;
      const webHits = web && (web.results || []).length
        ? `<div class="hit-meta">联网检索 · ${esc(web.backend || '—')} · ${esc(web.query || '')}</div>` +
          (web.results || [])
            .map(
              (item, index) => `
      <article class="hit">
        <div class="hit-head">
          <span class="rank">#${index + 1}</span>
          <span class="hit-cite">${esc(item.url || '')}</span>
        </div>
        <div class="hit-text">${esc(item.title || item.url || '')}</div>
        ${item.snippet ? `<div class="hit-meta">${esc(item.snippet)}</div>` : ''}
      </article>`
            )
            .join('')
        : '';
      const sourceLabel = data.answer_source === 'web_search' ? '联网搜索' : '本地知识库';
      const webError = data.web_search_error
        ? `<p class="callout">${esc(data.web_search_error)}</p>`
        : '';
      $('#answer-mode-hint').textContent = backendLabel;
      $('#answer-result').innerHTML = `
        <div class="answer-box">
          <p>${esc(data.answer || '（空回答）')}</p>
          <div class="citations">${citations || '<span>无引用</span>'}</div>
          <div class="hit-meta">回答后端：${esc(backendLabel)} · 答案来源：${esc(sourceLabel)}</div>
        </div>
        ${error}
        ${webError}
        ${webHits}
        ${hits}`;
      toast(
        data.web_search_error
          ? data.has_answer
            ? '联网检索失败，已返回本地结果'
            : '联网检索失败'
          : data.answer_source === 'web_search'
            ? `知识库未命中，已联网检索（${data.web_search?.backend || '—'}）`
            : data.llm_error
              ? '在线模型不可用，已回落离线回答'
              : data.has_answer
                ? '已生成带引用的回答'
                : '知识库中未找到相关内容',
        data.web_search_error || data.llm_error ? 'err' : 'ok'
      );
    }).catch(reason);
  }

  // -------------------------------------------------------------------- 记忆

  function memoryItemHtml(item, source) {
    const label = source === 'working' ? '工作记忆' : '长期记忆';
    const tagClass = source === 'working' ? 'work' : 'long';
    const type = item.memory_type || (item.metadata && item.metadata.memory_type) || '—';
    const meta = [
      item.memory_id || item.item_id,
      `重要度 ${Number(item.importance ?? 0).toFixed(1)}`,
      item.created_at || '',
      item.last_accessed_at ? `最近访问 ${item.last_accessed_at}` : '',
    ]
      .filter(Boolean)
      .map((part) => esc(part))
      .join(' · ');
    return `
      <article class="memory-item">
        <div class="memory-item-head">
          <span class="tag ${tagClass}">${label}</span>
          <span class="tag">${esc(type)}</span>
          ${item.score !== undefined ? `<span class="score">score ${Number(item.score).toFixed(4)}</span>` : ''}
        </div>
        <div class="memory-content">${esc(item.content || '')}</div>
        <div class="memory-meta">${meta}</div>
      </article>`;
  }

  function renderMemory(snapshot) {
    const stats = snapshot.stats || {};
    const longTerm = stats.long_term || {};
    const working = stats.working || {};
    $('#mem-count').textContent = longTerm.memories ?? 0;
    $('#mem-working').textContent = working.items ?? 0;
    $('#mem-dim').textContent = longTerm.dimension || 0;
    const types = longTerm.types || {};
    $('#mem-types').textContent =
      Object.keys(types).length
        ? Object.entries(types)
            .map(([key, value]) => `${key} ${value}`)
            .join(' · ')
        : '—';

    const longItems = (snapshot.long_term || []).filter((item) => (item.score || 0) >= 0);
    const workingItems = snapshot.working || [];
    const list = $('#memory-list');
    if (!longItems.length && !workingItems.length) {
      list.innerHTML = '<p class="empty">还没有记忆，先写入一条。</p>';
      $('#memory-list-count').textContent = '';
      return;
    }
    list.innerHTML = [
      ...longItems.map((item) => memoryItemHtml(item, 'long_term')),
      ...workingItems.map((item) => memoryItemHtml(item, 'working')),
    ].join('');
    $('#memory-list-count').textContent = `${longItems.length} 条长期 / ${workingItems.length} 条工作`;
  }

  async function loadMemory() {
    const snapshot = await api('/api/memory');
    renderMemory(snapshot);
  }

  async function addMemory(event) {
    event.preventDefault();
    const content = $('#memory-content').value.trim();
    if (!content) {
      toast('记忆内容不能为空', 'err');
      return;
    }
    await withBusy(event.submitter || $('#memory-add-form button'), '写入中…', async () => {
      const result = await api('/api/memory/add', {
        method: 'POST',
        body: {
          content,
          memory_type: $('#memory-type').value,
          importance: Number($('#memory-importance').value),
        },
      });
      renderMemory(result.snapshot);
      toast(`已写入记忆 ${result.memory_id}`);
      $('#memory-content').value = '';
    }).catch(reason);
  }

  async function recordTurn(event) {
    event.preventDefault();
    const question = $('#turn-question').value.trim();
    if (!question) {
      toast('用户输入不能为空', 'err');
      return;
    }
    await withBusy(event.submitter || $('#memory-turn-form button'), '写入中…', async () => {
      const result = await api('/api/memory/turn', {
        method: 'POST',
        body: { question, answer: $('#turn-answer').value.trim() },
      });
      renderMemory(result.snapshot);
      toast('已记录一轮对话为 episodic 记忆');
      $('#turn-question').value = '';
      $('#turn-answer').value = '';
    }).catch(reason);
  }

  async function searchMemory(event) {
    event.preventDefault();
    const query = $('#memory-query').value.trim();
    if (!query) {
      toast('请输入检索问题', 'err');
      return;
    }
    await withBusy(event.submitter || $('#memory-search-form button'), '检索中…', async () => {
      const data = await api('/api/memory/search', {
        method: 'POST',
        body: { query, top_k: Number($('#memory-topk').value) || 5 },
      });
      const entries = data.entries || [];
      $('#memory-search-results').innerHTML = entries.length
        ? entries.map((item) => memoryItemHtml(item, item.source || 'long_term')).join('')
        : '<p class="empty">没有召回记忆。</p>';
      toast(`召回 ${entries.length} 条记忆`);
    }).catch(reason);
  }

  async function buildContext(event) {
    event.preventDefault();
    const query = $('#context-query').value.trim();
    await withBusy(event.submitter || $('#memory-context-form button'), '构建中…', async () => {
      const data = await api('/api/memory/context', {
        method: 'POST',
        body: {
          query,
          max_tokens: Number($('#context-tokens').value) || 1500,
        },
      });
      const items = (data.items || [])
        .map((item) => memoryItemHtml(item, item.source || 'long_term'))
        .join('');
      $('#memory-context-result').innerHTML = `
        <article class="hit">
          <div class="hit-head"><span class="chip">context</span>
            <span class="muted">预算 ${esc(data.max_tokens)} tokens · ${(data.items || []).length} 条入选</span></div>
          <pre class="code-block">${esc(data.text || '（空上下文）')}</pre>
        </article>
        ${items}`;
      toast(`上下文已构建：${(data.items || []).length} 条`);
    }).catch(reason);
  }

  async function clearMemory() {
    if (!window.confirm('确定清空当前用户的工作记忆与长期记忆？')) return;
    try {
      const result = await api('/api/memory/clear', { method: 'POST', body: {} });
      renderMemory(result.snapshot);
      toast('记忆已清空');
    } catch (error) {
      reason(error);
    }
  }

  // ------------------------------------------------------------------ 三范式

  async function runParadigms(event) {
    event.preventDefault();
    const question = $('#paradigm-input').value.trim();
    const online = $('#paradigm-online').checked;
    if (question && !online) {
      toast('自定义问题需要勾选“使用在线 LLM”', 'err');
      return;
    }
    await withBusy(
      $('#run-paradigms'),
      online ? '运行中…（在线 LLM 可能较慢）' : '运行三范式…',
      async () => {
        const data = await api('/api/paradigms/run', {
          method: 'POST',
          body: { question, online },
        });
        const label = data.llm_backend === 'online'
          ? `在线 LLM${data.llm_model ? ` · ${data.llm_model}` : ''}`
          : '离线脚本 LLM';
        $('#paradigm-mode').textContent = label;
        const questionBox = $('#paradigm-question');
        questionBox.classList.remove('hidden');
        questionBox.textContent = `提问：${data.question || ''}（${label}）`;

        $('#paradigm-results').innerHTML = (data.rows || [])
          .map(
            (row) => `
          <div class="paradigm-block">
            <div class="paradigm-head"><span>${esc(row.paradigm)}</span><span class="muted">无记忆 vs 有记忆</span></div>
            <div class="paradigm-body">
              <div class="paradigm-cell"><h4>无记忆</h4><p>${esc(row.without_memory || '')}</p></div>
              <div class="paradigm-cell with"><h4>有记忆</h4><p>${esc(row.with_memory || '')}</p></div>
            </div>
          </div>`
          )
          .join('');

        $('#workflow-result').innerHTML = data.workflow_answer
          ? `<article class="answer-box"><h3 class="sub-title">工作流输出</h3><p>${esc(data.workflow_answer)}</p></article>`
          : '';
        toast('三范式对照完成');
      }
    ).catch(reason);
  }

  // -------------------------------------------------------------------- Demo

  function artifactsHtml(artifacts) {
    if (!artifacts || !artifacts.length) return '';
    return artifacts
      .map(
        (file) => `
        <a class="file-link" href="${esc(file.url)}" target="_blank" rel="noopener">
          ${esc(file.name)} <span class="size">${bytes(file.size)}</span>
        </a>`
      )
      .join('');
  }

  function renderRunList(runs) {
    const node = $('#demo-runs');
    if (!runs || !runs.length) {
      node.innerHTML = '<p class="empty">还没有运行记录。</p>';
      return;
    }
    node.innerHTML = runs
      .map(
        (run) => `
        <article class="run-item">
          <div class="run-id">${esc(run.run_id)}</div>
          <div class="muted">${esc(run.created_at)}</div>
          <div class="run-files">${artifactsHtml(run.files)}</div>
        </article>`
      )
      .join('');
  }

  async function loadRuns() {
    const data = await api('/api/runs');
    renderRunList(data.runs);
  }

  async function runDemo(event) {
    event.preventDefault();
    const button = event.submitter || $('#demo-form button');
    await withBusy(button, '运行中…（约 10–40 秒）', async () => {
      const data = await api('/api/demo/run', {
        method: 'POST',
        body: {
          embed_backend: $('#demo-embed').value,
          ingest: $('#demo-ingest').value.trim() || null,
          query: $('#demo-query').value.trim() || null,
          use_console_kb: $('#demo-use-kb').checked,
        },
      });
      const kbCount = data.console_kb_docs ?? 0;
      $('#demo-artifacts').innerHTML = `
        <article class="hit">
          <div class="hit-head"><span class="chip">${esc(data.run_id)}</span>
            <span class="muted">chunks ${esc((data.stats || {}).chunks ?? '—')} · 控制台文档 ${kbCount} 篇</span></div>
          <div class="run-files">${artifactsHtml(data.artifacts)}</div>
        </article>`;
      $('#demo-run-label').textContent = data.run_id;
      setMarkdown('#demo-report', data.report, '本次运行没有报告内容。');
      toast(`Demo 完成：${data.run_id}`);
      await loadRuns();
      markFresh('demo');
    }).catch(reason);
  }

  // -------------------------------------------------------------------- 评估

  function metricCard(result) {
    const recall = result.recall || {};
    const values = [
      `recall@1 <b>${Number(recall['recall@1'] ?? 0).toFixed(3)}</b>`,
      `recall@3 <b>${Number(recall['recall@3'] ?? 0).toFixed(3)}</b>`,
      `recall@5 <b>${Number(recall['recall@5'] ?? 0).toFixed(3)}</b>`,
      `MRR <b>${Number(result.mrr ?? 0).toFixed(3)}</b>`,
      `事实命中 <b>${Number(result.fact_rate ?? 0).toFixed(3)}</b>`,
      `引用覆盖 <b>${Number(result.citation_rate ?? 0).toFixed(3)}</b>`,
      `拒答率 <b>${Number(result.refusal_rate ?? 0).toFixed(3)}</b>`,
      `完整度 <b>${Number(result.completeness ?? 0).toFixed(2)}</b>`,
    ];
    return `
      <article class="metric-card">
        <span class="name">${esc(result.backend || '—')}</span>
        <span class="fingerprint">${esc(result.embedding_fingerprint || '—')}</span>
        <div class="metric-values">${values.map((item) => `<span>${item}</span>`).join('')}</div>
      </article>`;
  }

  async function loadEvalCases() {
    const data = await api('/api/eval/cases');
    const cases = data.cases || [];
    $('#eval-case-count').textContent = `${cases.length} 条样本`;
    $('#eval-case-rows').innerHTML = cases.length
      ? cases
          .map(
            (item) => `
          <tr>
            <td><span class="chip">${esc(item.id)}</span><div class="doc-source">${esc(item.city || '')}</div></td>
            <td>${esc(item.question)}</td>
            <td>${esc(item.gold_answer)}</td>
            <td class="mono wrap">${esc((item.gold_chunk_ids || []).join(', '))}</td>
          </tr>`
          )
          .join('')
      : '<tr><td colspan="4" class="empty">评估集为空。</td></tr>';
  }

  async function runEval() {
    await withBusy($('#run-eval'), '评估中…', async () => {
      const data = await api('/api/eval/run', {
        method: 'POST',
        body: {
          backend: $('#eval-backend').value,
          online: $('#eval-online').checked,
        },
      });
      const results = data.results || [];
      const skipped = data.skipped || [];
      $('#eval-metrics').innerHTML =
        results.map(metricCard).join('') +
        skipped
          .map((item) => `<article class="callout">跳过：${esc(item)}</article>`)
          .join('');
      setMarkdown('#eval-report', data.markdown, '评估报告为空。');
      toast(`评估完成：${results.map((item) => item.backend).join(', ') || '无结果'}`);
    }).catch(reason);
  }

  // -------------------------------------------------------------------- 测试

  function renderTests(tests) {
    const node = $('#test-list');
    if (!tests || !tests.length) {
      node.innerHTML = '<p class="empty">没有发现 tests/test_*.py。</p>';
      return;
    }
    node.innerHTML = tests
      .map(
        (item) => `
        <article class="test-item">
          <div>
            <div class="name">${esc(item.name)}</div>
            <div class="desc">${esc(item.description || item.label || '')}</div>
          </div>
          <button class="ghost-btn" data-test-target="${esc(item.target)}">运行</button>
        </article>`
      )
      .join('');
  }

  async function loadTests() {
    const data = await api('/api/tests');
    renderTests(data.tests);
  }

  function renderTestResult(result) {
    const summary = result.summary || {};
    const parts = Object.entries(summary).map(([key, value]) => `${value} ${key}`);
    $('#test-summary').textContent = `${result.target} · ${(result.duration_ms / 1000).toFixed(1)}s · ${
      parts.join(' / ') || `退出码 ${result.returncode}`
    }`;
    const output = $('#test-output');
    output.textContent = result.output || '（无输出）';
    output.scrollTop = output.scrollHeight;
  }

  async function runTests(target, button) {
    await withBusy(button, '运行中…', async () => {
      $('#test-summary').textContent = `正在运行 ${target}…`;
      const result = await api('/api/tests/run', {
        method: 'POST',
        body: { target },
      });
      renderTestResult(result);
      if (result.returncode === 0) {
        toast('测试通过');
      } else {
        toast('测试存在失败，请查看输出', 'err');
      }
    }).catch(reason);
  }

  // ------------------------------------------------------------------ 主题

  function initTheme() {
    let saved = null;
    try {
      saved = window.localStorage.getItem('agentflow-theme');
    } catch {
      saved = null;
    }
    if (saved === 'light' || saved === 'dark') {
      document.documentElement.dataset.theme = saved;
    }
    $('#theme-toggle').addEventListener('click', () => {
      const current = document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
      const next = current === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = next;
      try {
        window.localStorage.setItem('agentflow-theme', next);
      } catch {
        /* 忽略隐私模式下的写入失败 */
      }
    });
  }

  // ------------------------------------------------------------------ 绑定

  function bind() {
    $$('#tabs button').forEach((button) => {
      button.addEventListener('click', () => switchTab(button.dataset.tab));
    });

    $$('[data-goto]').forEach((button) => {
      button.addEventListener('click', () => {
        const target = button.dataset.goto;
        switchTab(target);
        const preset = button.dataset.presetQuery;
        if (!preset) return;
        if (target === 'memory') {
          $('#memory-query').value = preset;
          $('#context-query').value = preset;
          $('#memory-query').focus();
        } else {
          $('#search-query').value = preset;
          $('#answer-question').value = preset;
          $('#search-query').focus();
        }
      });
    });

    $('#upload-form').addEventListener('submit', ingestUpload);
    $('#settings-form').addEventListener('submit', applySettings);
    $('#url-form').addEventListener('submit', ingestUrl);
    $('#clear-docs').addEventListener('click', clearDocuments);
    $('#doc-rows').addEventListener('click', (event) => {
      const button = event.target.closest('[data-delete-doc]');
      if (button) deleteDocument(button.dataset.deleteDoc);
    });
    $('#search-form').addEventListener('submit', runSearch);
    $('#answer-form').addEventListener('submit', runAnswer);

    $('#memory-add-form').addEventListener('submit', addMemory);
    $('#memory-turn-form').addEventListener('submit', recordTurn);
    $('#memory-search-form').addEventListener('submit', searchMemory);
    $('#memory-context-form').addEventListener('submit', buildContext);
    $('#clear-memory').addEventListener('click', clearMemory);
    $('#memory-importance').addEventListener('input', (event) => {
      $('#importance-out').textContent = Number(event.target.value).toFixed(1);
    });

    $('#seed-toggle').addEventListener('change', toggleSeed);

    $('#paradigm-form').addEventListener('submit', runParadigms);

    $('#demo-form').addEventListener('submit', runDemo);
    $('#refresh-runs').addEventListener('click', () => loadRuns().catch(reason));

    $('#run-eval').addEventListener('click', runEval);

    $('#run-all-tests').addEventListener('click', (event) =>
      runTests('all', event.currentTarget)
    );
    $('#test-list').addEventListener('click', (event) => {
      const button = event.target.closest('[data-test-target]');
      if (button) runTests(button.dataset.testTarget, button);
    });
  }

  function start() {
    initTheme();
    bind();
    loadStatus().catch((error) => {
      setHealth(false, '未连接');
      reason(error);
    });
    state.loaded.add('overview');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
