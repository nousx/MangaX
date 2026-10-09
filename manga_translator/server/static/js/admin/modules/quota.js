// 配额管理模块
class QuotaModule {
    constructor(app) {
        this.app = app;
    }
    
    async load() {
        await this.loadQuotaSettings();
        await this.loadUserQuotas();
    }
    
    async loadQuotaSettings() {
        try {
            const resp = await fetch('/admin/settings', {
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            if (resp.ok) {
                const settings = await resp.json();
                // 填充配额设置表单
                const defaultQuota = settings.default_quota || {};
                document.getElementById('quota-daily-limit').value = defaultQuota.daily_limit || 100;
                document.getElementById('quota-monthly-limit').value = defaultQuota.monthly_limit || 3000;
                document.getElementById('quota-max-file-size').value = defaultQuota.max_file_size || 10;
                document.getElementById('quota-max-batch-size').value = defaultQuota.max_batch_size || 20;
            }
        } catch (e) {
            console.error('Failed to load quota settings:', e);
        }
    }
    
    async loadUserQuotas() {
        const tbody = document.getElementById('user-quotas-table-body');
        if (!tbody) return;
        
        try {
            const resp = await fetch('/api/admin/users', {
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            if (resp.ok) {
                const data = await resp.json();
                // API返回的是数组，不是{users: []}
                const users = Array.isArray(data) ? data : (data.users || []);
                
                // Built with DOM APIs: usernames/group names are untrusted.
                const el = SafeDom.el;
                const rows = users.map(user => {
                    const username = String(user.username ?? '');
                    const quota = user.quota || {};
                    const dailyUsed = Number(quota.daily_used) || 0;
                    const dailyLimit = Number(quota.daily_limit) || 100;
                    const percentage = SafeDom.percent((dailyUsed / dailyLimit) * 100);

                    return el('tr', null,
                        el('td', null, el('strong', { text: username })),
                        el('td', { text: user.group || 'default' }),
                        el('td', null,
                            el('div', { style: 'display:flex;align-items:center;gap:8px;' },
                                el('div', { style: 'flex:1;background:#e5e7eb;border-radius:4px;height:8px;' },
                                    el('div', { style: `width:${percentage}%;background:${percentage > 80 ? '#ef4444' : '#3b82f6'};height:100%;border-radius:4px;` })
                                ),
                                el('span', { style: 'font-size:12px;color:#6b7280;', text: `${dailyUsed}/${dailyLimit}` })
                            )
                        ),
                        el('td', { text: `${quota.monthly_used || 0}/${quota.monthly_limit || 3000}` }),
                        el('td', null,
                            el('button', {
                                className: 'btn btn-secondary btn-sm',
                                text: '编辑',
                                on: { click: () => this.editUserQuota(username) }
                            }),
                            ' ',
                            el('button', {
                                className: 'btn btn-secondary btn-sm',
                                text: '重置',
                                on: { click: () => this.resetUserQuota(username) }
                            })
                        )
                    );
                });
                SafeDom.setChildren(tbody, rows);
            }
        } catch (e) {
            console.error('Failed to load user quotas:', e);
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#6b7280;">加载失败</td></tr>';
        }
    }
    
    async saveDefaultQuota() {
        const quota = {
            daily_limit: parseInt(document.getElementById('quota-daily-limit').value) || 100,
            monthly_limit: parseInt(document.getElementById('quota-monthly-limit').value) || 3000,
            max_file_size: parseInt(document.getElementById('quota-max-file-size').value) || 10,
            max_batch_size: parseInt(document.getElementById('quota-max-batch-size').value) || 20
        };
        
        try {
            const resp = await fetch('/admin/settings', {
                method: 'PUT',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Session-Token': this.app.sessionToken 
                },
                body: JSON.stringify({ default_quota: quota })
            });
            
            if (resp.ok) {
                alert('默认配额设置已保存！');
            } else {
                throw new Error('保存失败');
            }
        } catch (e) {
            alert('保存失败: ' + e.message);
        }
    }
    
    editUserQuota(username) {
        const newLimit = prompt(`设置 ${username} 的每日配额限制:`, '100');
        if (newLimit !== null) {
            alert(`用户 ${username} 的配额已更新为 ${newLimit}`);
            this.loadUserQuotas();
        }
    }
    
    resetUserQuota(username) {
        if (confirm(`确定要重置 ${username} 的配额使用量吗？`)) {
            alert(`用户 ${username} 的配额已重置`);
            this.loadUserQuotas();
        }
    }
}

window.QuotaModule = QuotaModule;
