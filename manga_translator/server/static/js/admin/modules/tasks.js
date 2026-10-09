// 任务监控模块
class TasksModule {
    constructor(app) {
        this.app = app;
        this.autoRefresh = true;
        this.refreshInterval = null;
    }
    
    async load() {
        await this.loadTasks();
        this.startAutoRefresh();
    }
    
    async loadTasks() {
        const tbody = document.getElementById('all-tasks-table-body');
        if (!tbody) return;
        
        try {
            const resp = await fetch('/admin/tasks', {
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            
            if (resp.ok) {
                const tasks = await resp.json();
                this.renderTasks(tasks);
            }
        } catch (e) {
            console.error('Failed to load tasks:', e);
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#6b7280;">加载失败</td></tr>';
        }
    }
    
    renderTasks(tasks) {
        const tbody = document.getElementById('all-tasks-table-body');
        
        // 兼容数组和字典格式
        let taskList;
        if (Array.isArray(tasks)) {
            taskList = tasks.map(t => [t.task_id, t]);
        } else {
            taskList = Object.entries(tasks || {});
        }
        
        if (!taskList || taskList.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#6b7280;">暂无任务</td></tr>';
            document.getElementById('tasks-count').textContent = '0';
            return;
        }
        
        document.getElementById('tasks-count').textContent = taskList.length;
        
        // Built with DOM APIs: task ids, usernames, types and statuses are untrusted.
        const el = SafeDom.el;
        const rows = taskList.map(([rawId, task]) => {
            const id = String(rawId ?? '');
            const statusClass = {
                'pending': 'badge-warning',
                'queued': 'badge-warning',
                'processing': 'badge-info',
                'running': 'badge-info',
                'completed': 'badge-success',
                'failed': 'badge-danger',
                'cancelled': 'badge-secondary'
            }[task.status] || 'badge-info';
            const progress = SafeDom.percent(task.progress);
            const isActive = ['processing', 'pending', 'running', 'queued'].includes(task.status);

            return el('tr', null,
                el('td', null, el('code', {
                    style: 'background:#f3f4f6;padding:2px 6px;border-radius:4px;',
                    text: `${id.slice(0, 12)}...`
                })),
                el('td', { text: task.username || '-' }),
                el('td', { text: task.type || 'translate' }),
                el('td', null, el('span', { className: `badge ${statusClass}`, text: task.status || 'processing' })),
                el('td', null,
                    el('div', { style: 'display:flex;align-items:center;gap:8px;' },
                        el('div', { style: 'flex:1;max-width:100px;background:#e5e7eb;border-radius:4px;height:6px;' },
                            el('div', { style: `width:${progress}%;background:#3b82f6;height:100%;border-radius:4px;transition:width 0.3s;` })
                        ),
                        el('span', { style: 'font-size:12px;color:#6b7280;', text: `${progress}%` })
                    )
                ),
                el('td', { text: task.start_time ? new Date(task.start_time).toLocaleString() : '-' }),
                el('td', null, isActive
                    ? el('button', {
                        className: 'btn btn-danger btn-sm',
                        text: '取消',
                        on: { click: () => this.cancelTask(id) }
                    })
                    : el('button', {
                        className: 'btn btn-secondary btn-sm',
                        text: '详情',
                        on: { click: () => this.viewTaskDetail(id) }
                    })
                )
            );
        });
        SafeDom.setChildren(tbody, rows);
    }

    startAutoRefresh() {
        if (this.refreshInterval) clearInterval(this.refreshInterval);
        if (this.autoRefresh) {
            this.refreshInterval = setInterval(() => this.loadTasks(), 3000);
        }
    }
    
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }
    
    toggleAutoRefresh() {
        this.autoRefresh = !this.autoRefresh;
        const btn = document.getElementById('auto-refresh-btn');
        if (btn) {
            btn.textContent = this.autoRefresh ? '⏸ 暂停刷新' : '▶ 自动刷新';
            btn.classList.toggle('btn-primary', this.autoRefresh);
            btn.classList.toggle('btn-secondary', !this.autoRefresh);
        }
        if (this.autoRefresh) {
            this.startAutoRefresh();
        } else {
            this.stopAutoRefresh();
        }
    }
    
    async cancelTask(taskId) {
        if (!confirm('确定要取消此任务吗？')) return;
        
        try {
            const resp = await fetch(`/admin/tasks/${encodeURIComponent(taskId)}/cancel`, {
                method: 'POST',
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            
            if (resp.ok) {
                this.loadTasks();
            } else {
                throw new Error('取消失败');
            }
        } catch (e) {
            alert('取消任务失败: ' + e.message);
        }
    }
    
    async cancelAllTasks() {
        if (!confirm('确定要取消所有进行中的任务吗？')) return;
        
        try {
            const resp = await fetch('/admin/tasks/cancel-all', {
                method: 'POST',
                headers: { 'X-Session-Token': this.app.sessionToken }
            });
            
            if (resp.ok) {
                alert('所有任务已取消');
                this.loadTasks();
            }
        } catch (e) {
            alert('操作失败: ' + e.message);
        }
    }
    
    viewTaskDetail(taskId) {
        alert('任务详情: ' + taskId);
    }
}

window.TasksModule = TasksModule;
