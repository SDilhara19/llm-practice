css = '''
<style>
.chat-message {
    border-radius: 0.5rem;
    border: 2px solid green;
    padding: 10px;
    margin-bottom: 10px;
}
.chat-message.bot .avatar img,
.chat-message.user .avatar img {
    max-height: 78px;
    max-width: 78px;
    border-radius: 50%;
}
.chat-message.bot .avatar {
    float: left;
    margin-right: 10px;
}
.chat-message.user .avatar {
    float: right;
    margin-left: 10px;
}
</style>
'''

# Define the HTML template for bot messages
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="" alt="bot" />
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

# Define the HTML template for user messages
user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="" alt="user" />
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''