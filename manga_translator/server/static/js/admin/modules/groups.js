// 用户组管理模块 - 使用通用权限编辑器
class GroupsModule {
    constructor(app) {
        this.app = app;
        this.currentEditGroup = null;
        this.presets = [];
    }

    async load() {
        await Promise.all([
            this.loadGroups(),
            this.loadPresets()
        ]);
    }

    async loadPresets() {
        try {
            const resp = await fetch('/api/admin/presets', {
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            if (resp.ok) {
                const data = await resp.json();
                this.presets = data.presets || [];
            }
        } catch (e) {
            console.error('Failed to load presets:', e);
        }
    }

    async loadGroups() {
        try {
            const resp = await fetch('/api/admin/groups', {
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            if (resp.ok) {
                const data = await resp.json();
                this.renderGroupsTable(data.groups || []);
            }
        } catch (e) {
            console.error('Failed to load groups:', e);
        }
    }

    renderGroupsTable(groups) {
        const tbody = document.getElementById('groups-table-body');
        if (!tbody) return;
        if (groups.length === 0) {
            SafeDom.setChildren(tbody, SafeDom.messageRow(6, '暂无用户组'));
            return;
        }
        // Built with DOM APIs: group ids/names/descriptions are untrusted.
        const el = SafeDom.el;
        const rows = groups.map(group => {
            const groupId = String(group.id ?? '');
            const presetName = group.default_preset_id
                ? (this.presets.find(p => p.id === group.default_preset_id)?.name || group.default_preset_id)
                : '服务器默认';
            const isBuiltin = ['admin', 'default', 'guest'].includes(groupId);
            return el('tr', null,
                el('td', null,
                    el('strong', { text: group.name }),
                    el('br'),
                    el('small', { style: 'color:#6b7280;', text: groupId })
                ),
                el('td', { text: group.description || '-' }),
                el('td', { text: group.member_count || 0 }),
                el('td', null, el('span', {
                    className: 'badge badge-info',
                    title: 'API密钥预设',
                    text: presetName
                })),
                el('td', null, el('span', {
                    className: `badge ${group.is_default ? 'badge-success' : 'badge-secondary'}`,
                    text: group.is_default ? '是' : '否'
                })),
                el('td', null,
                    el('button', {
                        className: 'btn btn-primary btn-sm',
                        text: '编辑',
                        on: { click: () => this.openEditModal(groupId) }
                    }),
                    ' ',
                    !isBuiltin && el('button', {
                        className: 'btn btn-danger btn-sm',
                        text: '删除',
                        on: { click: () => this.deleteGroup(groupId) }
                    })
                )
            );
        });
        SafeDom.setChildren(tbody, rows);
    }

    async openEditModal(groupId) {
        try {
            const resp = await fetch(`/api/admin/groups/${encodeURIComponent(groupId)}`, {
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            if (!resp.ok) throw new Error('获取用户组失败');
            const data = await resp.json();
            this.currentEditGroup = data.group || data;

            // 直接使用权限编辑器
            const editor = new PermissionEditor({
                title: `编辑用户组: ${this.currentEditGroup.name}`,
                mode: 'group',
                sessionToken: this.app.sessionToken,
                onSave: (config) => this.saveGroupConfig(config),
                onCancel: () => { this.currentEditGroup = null; }
            });

            // 传递用户组的参数配置
            const groupConfig = {
                ...this.currentEditGroup.parameter_config || {},
                allowed_translators: this.currentEditGroup.allowed_translators || ['*'],
                denied_translators: this.currentEditGroup.denied_translators || [],
                allowed_ocr: this.currentEditGroup.allowed_ocr || ['*'],
                denied_ocr: this.currentEditGroup.denied_ocr || [],
                allowed_colorizers: this.currentEditGroup.allowed_colorizers || ['*'],
                denied_colorizers: this.currentEditGroup.denied_colorizers || [],
                allowed_renderers: this.currentEditGroup.allowed_renderers || ['*'],
                denied_renderers: this.currentEditGroup.denied_renderers || [],
                allowed_workflows: this.currentEditGroup.allowed_workflows || ['*'],
                denied_workflows: this.currentEditGroup.denied_workflows || [],
                visible_presets: this.currentEditGroup.visible_presets || []
            };
            // 添加预设ID到配置中
            if (this.currentEditGroup.default_preset_id) {
                groupConfig._meta = groupConfig._meta || {};
                groupConfig._meta.default_preset_id = this.currentEditGroup.default_preset_id;
            }
            await editor.show(groupConfig);
        } catch (e) {
            alert('获取用户组信息失败: ' + e.message);
        }
    }

    async saveGroupConfig(config) {
        if (!this.currentEditGroup) return;

        // 提取预设ID
        const defaultPresetId = config._meta?.default_preset_id || null;
        // 提取翻译器配置
        const allowedTranslators = config.allowed_translators || ['*'];
        const deniedTranslators = config.denied_translators || [];
        const allowedOcr = config.allowed_ocr || ['*'];
        const deniedOcr = config.denied_ocr || [];
        const allowedColorizers = config.allowed_colorizers || ['*'];
        const deniedColorizers = config.denied_colorizers || [];
        const allowedRenderers = config.allowed_renderers || ['*'];
        const deniedRenderers = config.denied_renderers || [];
        // 提取工作流配置
        const allowedWorkflows = config.allowed_workflows || ['*'];
        const deniedWorkflows = config.denied_workflows || [];
        // 提取可见预设配置
        const visiblePresets = config.visible_presets || [];
        
        // 从config中移除非参数配置字段
        const paramConfig = { ...config };
        delete paramConfig._meta;
        delete paramConfig.allowed_translators;
        delete paramConfig.denied_translators;
        delete paramConfig.allowed_ocr;
        delete paramConfig.denied_ocr;
        delete paramConfig.allowed_colorizers;
        delete paramConfig.denied_colorizers;
        delete paramConfig.allowed_renderers;
        delete paramConfig.denied_renderers;
        delete paramConfig.allowed_workflows;
        delete paramConfig.denied_workflows;
        delete paramConfig.visible_presets;

        try {
            const resp = await fetch(`/api/admin/groups/${encodeURIComponent(this.currentEditGroup.id)}/config`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-Token': this.app.sessionToken
                },
                body: JSON.stringify({ 
                    parameter_config: paramConfig,
                    default_preset_id: defaultPresetId,
                    allowed_translators: allowedTranslators,
                    denied_translators: deniedTranslators,
                    allowed_ocr: allowedOcr,
                    denied_ocr: deniedOcr,
                    allowed_colorizers: allowedColorizers,
                    denied_colorizers: deniedColorizers,
                    allowed_renderers: allowedRenderers,
                    denied_renderers: deniedRenderers,
                    allowed_workflows: allowedWorkflows,
                    denied_workflows: deniedWorkflows,
                    visible_presets: visiblePresets
                })
            });

            if (resp.ok) {
                alert('用户组配置已保存！');
                this.loadGroups();
            } else {
                throw new Error('保存失败');
            }
        } catch (e) {
            alert('保存失败: ' + e.message);
        }
        this.currentEditGroup = null;
    }

    async deleteGroup(groupId) {
        if (!confirm(`确定删除用户组 "${groupId}"? 组内用户将被移至默认组。`)) return;

        try {
            const resp = await fetch(`/api/admin/groups/${encodeURIComponent(groupId)}`, {
                method: 'DELETE',
                headers: { 'X-Session-Token': this.app.sessionToken }
            });

            if (resp.ok) {
                alert('用户组已删除');
                this.loadGroups();
            } else {
                const err = await resp.json();
                alert('删除失败: ' + (err.detail?.message || '未知错误'));
            }
        } catch (e) {
            alert('删除失败: ' + e.message);
        }
    }

    showCreateGroupModal() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal" style="max-width:500px;background:#fff;border-radius:8px;box-shadow:0 20px 60px rgba(0,0,0,0.3);">
                <div class="modal-header" style="background:linear-gradient(135deg, #10b981 0%, #059669 100%);color:#fff;border-radius:8px 8px 0 0;">
                    <h3 style="margin:0;">➕ 创建用户组</h3>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()" style="color:#fff;">×</button>
                </div>
                <div class="modal-body" style="padding:24px;background:#fff;">
                    <div class="form-group">
                        <label class="form-label">用户组ID <span style="color:#ef4444;">*</span></label>
                        <input type="text" class="form-input" id="new-group-id" placeholder="例如: vip, premium">
                        <small style="color:#6b7280;">只能包含字母、数字和下划线</small>
                    </div>
                    <div class="form-group">
                        <label class="form-label">显示名称 <span style="color:#ef4444;">*</span></label>
                        <input type="text" class="form-input" id="new-group-name" placeholder="例如: VIP用户组">
                    </div>
                    <div class="form-group">
                        <label class="form-label">描述</label>
                        <textarea class="form-input" id="new-group-description" rows="2" placeholder="用户组的用途说明"></textarea>
                    </div>
                    <div class="form-group">
                        <label class="form-label">默认API密钥预设</label>
                        <select class="form-select" id="new-group-preset" style="width:100%;">
                            <option value="">使用服务器默认配置</option>
                        </select>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="this.closest('.modal-overlay').remove()">取消</button>
                    <button class="btn btn-primary" onclick="groupsModule.createGroup()" style="background:linear-gradient(135deg, #10b981 0%, #059669 100%);border:none;">✅ 创建</button>
                </div>
            </div>
        `;
        const presetSelect = modal.querySelector('#new-group-preset');
        for (const preset of this.presets) {
            const option = document.createElement('option');
            option.value = preset.id;
            option.textContent = preset.name;
            presetSelect.appendChild(option);
        }
        document.body.appendChild(modal);
    }

    async createGroup() {
        const groupId = document.getElementById('new-group-id')?.value?.trim();
        const name = document.getElementById('new-group-name')?.value?.trim();
        const description = document.getElementById('new-group-description')?.value?.trim();
        const defaultPresetId = document.getElementById('new-group-preset')?.value || null;

        if (!groupId || !name) {
            alert('请填写用户组ID和名称');
            return;
        }

        if (!/^[a-zA-Z0-9_]+$/.test(groupId)) {
            alert('用户组ID只能包含字母、数字和下划线');
            return;
        }

        try {
            const resp = await fetch('/api/admin/groups', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-Token': this.app.sessionToken
                },
                body: JSON.stringify({
                    group_id: groupId,
                    name: name,
                    description: description,
                    default_preset_id: defaultPresetId
                })
            });

            if (resp.ok) {
                document.querySelector('.modal-overlay')?.remove();
                alert('用户组创建成功！');
                this.loadGroups();
            } else {
                const err = await resp.json();
                alert('创建失败: ' + (err.detail?.message || '未知错误'));
            }
        } catch (e) {
            alert('创建失败: ' + e.message);
        }
    }

    escapeHtml(str) {
        return SafeDom.escapeHtml(str);
    }
}

window.GroupsModule = GroupsModule;
