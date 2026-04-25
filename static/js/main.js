// 
// BreastCancerAI - Main JavaScript File
// Optimized & Error-Free Version
// 

(function() {
    'use strict';
    
    // انتظار تحميل الصفحة بالكامل
    document.addEventListener('DOMContentLoaded', function() {
        console.log('🚀 BreastCancerAI initialized');
        
        // 
        // 1. MOBILE MENU SYSTEM
        // 
        function initMobileMenu() {
            const mobileMenuBtn = document.getElementById('mobileMenuBtn');
            const mobileMenu = document.getElementById('mobileMenu');
            
            if (!mobileMenuBtn || !mobileMenu) {
                console.log('📱 Mobile menu not available on this page');
                return;
            }
            
            console.log('📱 Mobile menu initialized');
            
            // Toggle menu on button click
            mobileMenuBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                mobileMenu.classList.toggle('active');
            });
            
            // Close menu when clicking on any link inside
            const mobileLinks = mobileMenu.querySelectorAll('a');
            mobileLinks.forEach(link => {
                link.addEventListener('click', function() {
                    mobileMenu.classList.remove('active');
                });
            });
            
            // Close menu when clicking outside
            document.addEventListener('click', function(e) {
                if (mobileMenu.classList.contains('active')) {
                    if (!mobileMenuBtn.contains(e.target) && !mobileMenu.contains(e.target)) {
                        mobileMenu.classList.remove('active');
                    }
                }
            });
        }
        
        // 
        // 2. USER DROPDOWN MENU (FIXED)
        // 
        function initUserDropdown() {
            const userMenuBtn = document.getElementById('userMenuBtn');
            const userDropdown = document.getElementById('userDropdown');
            
            if (!userMenuBtn || !userDropdown) {
                console.log('👤 User dropdown not available on this page');
                return;
            }
            
            console.log('👤 User dropdown initialized');
            
            // Toggle dropdown
            userMenuBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                e.preventDefault();
                userDropdown.classList.toggle('show');
            });
            
            // Close dropdown when clicking outside
            document.addEventListener('click', function(e) {
                if (userDropdown.classList.contains('show')) {
                    if (!userMenuBtn.contains(e.target)) {
                        userDropdown.classList.remove('show');
                    }
                }
            });
            
            // Prevent dropdown from closing when clicking inside
            userDropdown.addEventListener('click', function(e) {
                e.stopPropagation();
            });
        }
        
        // 
        // 3. FLASH MESSAGES AUTO-HIDE
        // 
        function initFlashMessages() {
            const flashMessages = document.querySelectorAll('.flash-message');
            
            if (flashMessages.length === 0) {
                return;
            }
            
            console.log(`💬 Auto-hiding ${flashMessages.length} flash message(s)`);
            
            setTimeout(() => {
                flashMessages.forEach(msg => {
                    msg.style.transition = 'opacity 0.5s ease';
                    msg.style.opacity = '0';
                    setTimeout(() => {
                        if (msg.parentNode) {
                            msg.remove();
                        }
                    }, 500);
                });
            }, 4000);
        }
        
        // 
        // 4. VIDEO CONTROLS (Only on index page)
        // 
        function initVideoControls() {
            const videoIframe = document.getElementById('youtubeVideo');
            
            if (!videoIframe) {
                console.log('🎬 Video player not available on this page');
                return;
            }
            
            console.log('🎬 Video controls initialized');
            
            const playBtn = document.getElementById('playVideo');
            const pauseBtn = document.getElementById('pauseVideo');
            const restartBtn = document.getElementById('restartVideo');
            
            if (playBtn) {
                playBtn.addEventListener('click', () => {
                    videoIframe.contentWindow.postMessage('{"event":"command","func":"playVideo","args":""}', '*');
                });
            }
            
            if (pauseBtn) {
                pauseBtn.addEventListener('click', () => {
                    videoIframe.contentWindow.postMessage('{"event":"command","func":"pauseVideo","args":""}', '*');
                });
            }
            
            if (restartBtn) {
                restartBtn.addEventListener('click', () => {
                    videoIframe.contentWindow.postMessage('{"event":"command","func":"seekTo","args":[0, true]}', '*');
                });
            }
        }
        
        // 
        // 5. CONTACT FORM HANDLER (Only on contact page)
        // 
        function initContactForm() {
            const contactForm = document.getElementById('contactForm');
            
            if (!contactForm) {
                console.log('📝 Contact form not available on this page');
                return;
            }
            
            console.log('📝 Contact form initialized');
            
            contactForm.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                // جمع البيانات
                const formData = new FormData(contactForm);
                const data = {};
                formData.forEach((value, key) => {
                    data[key] = value;
                });
                
                const submitBtn = contactForm.querySelector('button[type="submit"], .btn-submit');
                if (!submitBtn) return;
                
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
                submitBtn.disabled = true;
                
                try {
                    const response = await fetch('/send-message', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        alert('✅ Message sent successfully! We will contact you soon.');
                        contactForm.reset();
                    } else {
                        alert('❌ Error sending message. Please try again.');
                    }
                } catch (error) {
                    console.error('Contact form error:', error);
                    alert('❌ An error occurred. Please try again later.');
                } finally {
                    submitBtn.innerHTML = originalText;
                    submitBtn.disabled = false;
                }
            });
        }
        
        // 
        // 6. THEME TOGGLE (إذا لم يكن موجوداً في base.html)
        // 
        function initThemeToggle() {
            const themeToggle = document.getElementById('themeToggle');
            
            if (!themeToggle) {
                return;
            }
            
            const htmlElement = document.documentElement;
            const savedTheme = localStorage.getItem('theme') || 'light';
            htmlElement.setAttribute('data-theme', savedTheme);
            
            // تحديث الأيقونة
            updateThemeIcon(savedTheme);
            
            themeToggle.addEventListener('click', function() {
                const currentTheme = htmlElement.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                htmlElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                updateThemeIcon(newTheme);
            });
        }
        
        function updateThemeIcon(theme) {
            const themeToggle = document.getElementById('themeToggle');
            if (!themeToggle) return;
            
            const icon = themeToggle.querySelector('i');
            if (icon) {
                if (theme === 'dark') {
                    icon.className = 'fas fa-moon';
                } else {
                    icon.className = 'fas fa-sun';
                }
            }
        }
        
        // 
        // INITIALIZE ALL SYSTEMS
        // 
        initMobileMenu();
        initUserDropdown();
        initFlashMessages();
        initVideoControls();
        initContactForm();
        initThemeToggle();
        
        console.log('✅ All systems ready!');
    });
    
})();
